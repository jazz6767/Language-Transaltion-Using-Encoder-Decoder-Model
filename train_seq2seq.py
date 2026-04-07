import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
BASE_DIR = Path(__file__).resolve().parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize(text: str) -> List[str]:
    return text.strip().lower().split()


class Vocabulary:
    def __init__(self) -> None:
        self.token_to_idx: Dict[str, int] = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
        self.idx_to_token: Dict[int, str] = {idx: token for idx, token in enumerate(SPECIAL_TOKENS)}

    def add_sentence(self, sentence: str) -> None:
        for token in tokenize(sentence):
            self.add_token(token)

    def add_token(self, token: str) -> None:
        if token not in self.token_to_idx:
            index = len(self.token_to_idx)
            self.token_to_idx[token] = index
            self.idx_to_token[index] = token

    def encode(self, sentence: str) -> List[int]:
        tokens = [SOS_TOKEN] + tokenize(sentence) + [EOS_TOKEN]
        return [self.token_to_idx.get(token, self.token_to_idx[UNK_TOKEN]) for token in tokens]

    def decode(self, indices: List[int]) -> str:
        tokens: List[str] = []
        for idx in indices:
            token = self.idx_to_token.get(idx, UNK_TOKEN)
            if token == EOS_TOKEN:
                break
            if token not in {PAD_TOKEN, SOS_TOKEN}:
                tokens.append(token)
        return " ".join(tokens)

    def to_dict(self) -> Dict[str, Dict[str, int]]:
        return {
            "token_to_idx": self.token_to_idx,
            "idx_to_token": {str(k): v for k, v in self.idx_to_token.items()},
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Dict[str, int]]) -> "Vocabulary":
        vocab = cls()
        vocab.token_to_idx = payload["token_to_idx"]
        vocab.idx_to_token = {int(k): v for k, v in payload["idx_to_token"].items()}
        return vocab

    def __len__(self) -> int:
        return len(self.token_to_idx)


def read_parallel_data(data_path: Path, max_samples: Optional[int] = None) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with data_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            if "\t" not in line:
                raise ValueError(f"Invalid line {line_number}: expected tab-separated sentence pair.")
            columns = line.split("\t")
            if len(columns) < 2:
                raise ValueError(f"Invalid line {line_number}: expected at least two tab-separated columns.")
            source, target = columns[0], columns[1]
            pairs.append((source.strip(), target.strip()))
            if max_samples is not None and len(pairs) >= max_samples:
                break
    if not pairs:
        raise ValueError("Dataset is empty.")
    return pairs


class TranslationDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], source_vocab: Vocabulary, target_vocab: Vocabulary) -> None:
        self.source_sequences = [source_vocab.encode(src) for src, _ in pairs]
        self.target_sequences = [target_vocab.encode(tgt) for _, tgt in pairs]

    def __len__(self) -> int:
        return len(self.source_sequences)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        return self.source_sequences[index], self.target_sequences[index]


def collate_batch(batch: List[Tuple[List[int], List[int]]]) -> Tuple[torch.Tensor, torch.Tensor]:
    source_batch, target_batch = zip(*batch)
    source_max_len = max(len(seq) for seq in source_batch)
    target_max_len = max(len(seq) for seq in target_batch)
    pad_idx = SPECIAL_TOKENS.index(PAD_TOKEN)

    source_tensor = torch.full((len(batch), source_max_len), pad_idx, dtype=torch.long)
    target_tensor = torch.full((len(batch), target_max_len), pad_idx, dtype=torch.long)

    for row, sequence in enumerate(source_batch):
        source_tensor[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    for row, sequence in enumerate(target_batch):
        target_tensor[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)

    return source_tensor, target_tensor


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, source_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(source_tokens))
        _, (hidden, cell) = self.lstm(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        input_tokens: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(input_tokens.unsqueeze(1)))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        logits = self.output(output.squeeze(1))
        return logits, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source_tokens: torch.Tensor, target_tokens: torch.Tensor, teacher_forcing_ratio: float) -> torch.Tensor:
        batch_size, target_length = target_tokens.shape
        vocab_size = self.decoder.output.out_features
        outputs = torch.zeros(batch_size, target_length, vocab_size, device=self.device)

        hidden, cell = self.encoder(source_tokens)
        input_tokens = target_tokens[:, 0]

        for step in range(1, target_length):
            logits, hidden, cell = self.decoder(input_tokens, hidden, cell)
            outputs[:, step] = logits

            use_teacher_forcing = random.random() < teacher_forcing_ratio
            predicted_tokens = logits.argmax(dim=1)
            input_tokens = target_tokens[:, step] if use_teacher_forcing else predicted_tokens

        return outputs

    def translate(self, source_tokens: torch.Tensor, sos_idx: int, eos_idx: int, max_len: int = 20) -> List[int]:
        self.eval()
        with torch.no_grad():
            hidden, cell = self.encoder(source_tokens)
            input_token = torch.tensor([sos_idx], dtype=torch.long, device=self.device)
            generated: List[int] = []

            for _ in range(max_len):
                logits, hidden, cell = self.decoder(input_token, hidden, cell)
                next_token = int(logits.argmax(dim=1).item())
                if next_token == eos_idx:
                    break
                generated.append(next_token)
                input_token = torch.tensor([next_token], dtype=torch.long, device=self.device)

        return generated


@dataclass
class TrainingConfig:
    data_path: str
    source_lang: str
    target_lang: str
    device: str = "auto"
    max_samples: Optional[int] = None
    embedding_dim: int = 128
    hidden_dim: int = 256
    dropout: float = 0.2
    batch_size: int = 8
    epochs: int = 300
    learning_rate: float = 0.001
    teacher_forcing_ratio: float = 0.5
    seed: int = 42


def build_vocabs(pairs: List[Tuple[str, str]]) -> Tuple[Vocabulary, Vocabulary]:
    source_vocab = Vocabulary()
    target_vocab = Vocabulary()
    for source_sentence, target_sentence in pairs:
        source_vocab.add_sentence(source_sentence)
        target_vocab.add_sentence(target_sentence)
    return source_vocab, target_vocab


def resolve_device(device_preference: str) -> torch.device:
    preference = device_preference.lower()
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no GPU is available to PyTorch.")
        return torch.device("cuda")
    if preference == "cpu":
        return torch.device("cpu")
    raise ValueError("Invalid device. Use one of: auto, cpu, cuda.")


def train_model(config: TrainingConfig) -> None:
    set_seed(config.seed)
    device = resolve_device(config.device)
    data_path = Path(config.data_path)
    print(f"Loading dataset from: {data_path}")
    pairs = read_parallel_data(data_path, max_samples=config.max_samples)
    print(f"Loaded {len(pairs)} sentence pairs")
    source_vocab, target_vocab = build_vocabs(pairs)
    print(f"Source vocab size: {len(source_vocab)} | Target vocab size: {len(target_vocab)}")
    dataset = TranslationDataset(pairs, source_vocab, target_vocab)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_batch)
    print(f"Training on device: {device}")
    print(f"Epochs: {config.epochs} | Batch size: {config.batch_size} | Batches per epoch: {len(loader)}")

    encoder = Encoder(len(source_vocab), config.embedding_dim, config.hidden_dim, config.dropout)
    decoder = Decoder(len(target_vocab), config.embedding_dim, config.hidden_dim, config.dropout)
    model = Seq2Seq(encoder, decoder, device).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=target_vocab.token_to_idx[PAD_TOKEN])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        print(f"Starting epoch {epoch}/{config.epochs}...")

        for source_batch, target_batch in loader:
            source_batch = source_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()
            outputs = model(source_batch, target_batch, config.teacher_forcing_ratio)
            logits = outputs[:, 1:].reshape(-1, outputs.size(-1))
            targets = target_batch[:, 1:].reshape(-1)

            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, len(loader))
        print(f"Epoch {epoch:03d}/{config.epochs} | Loss: {avg_loss:.4f}")

    artifacts_dir = BASE_DIR / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    torch.save(model.state_dict(), artifacts_dir / "model.pt")
    with (artifacts_dir / "vocab.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_vocab": source_vocab.to_dict(),
                "target_vocab": target_vocab.to_dict(),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    with (artifacts_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)

    print(f"Training complete. Saved model artifacts to: {artifacts_dir.resolve()}")


def load_artifacts() -> Tuple[Seq2Seq, Vocabulary, Vocabulary, TrainingConfig, torch.device]:
    artifacts_dir = BASE_DIR / "artifacts"
    config_payload = json.loads((artifacts_dir / "config.json").read_text(encoding="utf-8"))
    vocab_payload = json.loads((artifacts_dir / "vocab.json").read_text(encoding="utf-8"))
    config = TrainingConfig(**config_payload)

    source_vocab = Vocabulary.from_dict(vocab_payload["source_vocab"])
    target_vocab = Vocabulary.from_dict(vocab_payload["target_vocab"])
    device = resolve_device(config.device)

    encoder = Encoder(len(source_vocab), config.embedding_dim, config.hidden_dim, config.dropout)
    decoder = Decoder(len(target_vocab), config.embedding_dim, config.hidden_dim, config.dropout)
    model = Seq2Seq(encoder, decoder, device).to(device)
    state_dict = torch.load(artifacts_dir / "model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, source_vocab, target_vocab, config, device


def translate_sentence(model: Seq2Seq, source_vocab: Vocabulary, target_vocab: Vocabulary, device: torch.device, sentence: str) -> str:
    encoded = source_vocab.encode(sentence)
    source_tensor = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    generated = model.translate(
        source_tensor,
        sos_idx=target_vocab.token_to_idx[SOS_TOKEN],
        eos_idx=target_vocab.token_to_idx[EOS_TOKEN],
    )
    return target_vocab.decode(generated)


def interactive_translate() -> None:
    model, source_vocab, target_vocab, config, device = load_artifacts()
    print(f"Interactive translation ready: {config.source_lang} -> {config.target_lang}")
    print("Type a sentence and press Enter. Type 'quit' to exit.")

    while True:
        sentence = input("> ").strip()
        if sentence.lower() in {"quit", "exit"}:
            break
        if not sentence:
            continue
        translation = translate_sentence(model, source_vocab, target_vocab, device, sentence)
        print(f"Translation: {translation}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or run an LSTM Seq2Seq translation model.")
    parser.add_argument("--mode", choices=["train", "translate"], default="train")
    parser.add_argument("--data_path", default="data/sample_en_fr.tsv")
    parser.add_argument("--source_lang", default="en")
    parser.add_argument("--target_lang", default="fr")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "translate":
        interactive_translate()
        return

    config = TrainingConfig(
        data_path=args.data_path,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        device=args.device,
        max_samples=args.max_samples,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        teacher_forcing_ratio=args.teacher_forcing_ratio,
        seed=args.seed,
    )
    train_model(config)


if __name__ == "__main__":
    main()
