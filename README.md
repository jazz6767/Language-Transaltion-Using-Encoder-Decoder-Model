# Language Translation (Encoder-Decoder)

This project implements a Seq2Seq language translation model using an LSTM-based encoder-decoder in PyTorch.

Supported examples:

- English -> French
- English -> Tamil

## Project Structure

- `train_seq2seq.py` - training, evaluation, and interactive translation
- `app.py` - Flask frontend for browser-based translation
- `templates/index.html` - main frontend page
- `static/styles.css` - frontend styling
- `requirements.txt` - Python dependencies
- `data/sample_en_fr.tsv` - tiny English-French sample dataset
- `data/sample_en_ta.tsv` - tiny English-Tamil sample dataset

## Dataset Format

Use a tab-separated file with one sentence pair per line:

```text
english sentence<TAB>target sentence
```

## Setup

```bash
pip install -r requirements.txt
```

## Train

English -> French:

```bash
python train_seq2seq.py --data_path data/sample_en_fr.tsv --source_lang en --target_lang fr --epochs 300
```

Train on a larger French dataset file such as `C:/Users/admin/OneDrive/Desktop/fra.txt`:

```bash
python train_seq2seq.py --data_path "C:/Users/admin/OneDrive/Desktop/fra.txt" --source_lang en --target_lang fr --max_samples 50000 --epochs 15 --batch_size 64
```

Force GPU training when CUDA is available:

```bash
python train_seq2seq.py --data_path "C:/Users/admin/OneDrive/Desktop/fra.txt" --source_lang en --target_lang fr --max_samples 20000 --epochs 8 --batch_size 64 --device cuda
```

English -> Tamil:

```bash
python train_seq2seq.py --data_path data/sample_en_ta.tsv --source_lang en --target_lang ta --epochs 300
```

## Translate

After training, the script saves:

- `artifacts/config.json`
- `artifacts/model.pt`
- `artifacts/vocab.json`

Run interactive inference:

```bash
python train_seq2seq.py --mode translate
```

## Frontend

Start the web app:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

The frontend uses the latest files in `artifacts/`. If those files are missing, train the model first.

## Large Dataset Tips

- `fra.txt` contains extra metadata columns, and the trainer automatically uses only the first two columns.
- Start with `--max_samples 50000` for a manageable first run.
- After training on `fra.txt`, restart the Flask app so it picks up the new artifacts.
- Use `--device cuda` to force GPU training, `--device cpu` to force CPU, or leave the default `auto`.

## Notes

- The included datasets are intentionally tiny, so translations are only for demonstration.
- For real performance, replace the sample dataset with a larger parallel corpus.
- The model uses greedy decoding for simplicity.
