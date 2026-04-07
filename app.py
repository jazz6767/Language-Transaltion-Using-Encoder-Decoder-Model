from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, render_template, request

from train_seq2seq import load_artifacts, read_parallel_data, translate_sentence


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

app = Flask(__name__)


@lru_cache(maxsize=1)
def get_model_bundle() -> Dict[str, Any]:
    if not ARTIFACTS_DIR.exists():
        raise FileNotFoundError("No trained model artifacts were found. Train the model first.")

    model, source_vocab, target_vocab, config, device = load_artifacts()
    pairs = read_parallel_data(BASE_DIR / config.data_path)
    phrasebook = {source.strip().lower(): target for source, target in pairs}
    return {
        "model": model,
        "source_vocab": source_vocab,
        "target_vocab": target_vocab,
        "config": config,
        "device": device,
        "phrasebook": phrasebook,
        "is_demo_dataset": len(phrasebook) <= 100,
    }


def best_phrasebook_match(sentence: str, phrasebook: Dict[str, str]) -> Dict[str, Any] | None:
    normalized = sentence.strip().lower()
    if normalized in phrasebook:
        return {
            "translation": phrasebook[normalized],
            "mode": "phrasebook",
            "matched_source": normalized,
            "score": 1.0,
        }

    best_source = None
    best_score = 0.0
    for source in phrasebook:
        score = SequenceMatcher(None, normalized, source).ratio()
        if score > best_score:
            best_score = score
            best_source = source

    if best_source and best_score >= 0.72:
        return {
            "translation": phrasebook[best_source],
            "mode": "closest_phrase",
            "matched_source": best_source,
            "score": round(best_score, 2),
        }

    return None


@app.get("/")
def index() -> str:
    context: Dict[str, Any] = {
        "ready": False,
        "source_lang": "en",
        "target_lang": "fr",
        "error": None,
        "supported_examples": [],
    }
    try:
        bundle = get_model_bundle()
        context["ready"] = True
        context["source_lang"] = bundle["config"].source_lang
        context["target_lang"] = bundle["config"].target_lang
        context["supported_examples"] = list(bundle["phrasebook"].keys())
    except Exception as exc:
        context["error"] = str(exc)

    return render_template("index.html", **context)


@app.post("/translate")
def translate() -> Any:
    payload = request.get_json(silent=True) or {}
    sentence = str(payload.get("sentence", "")).strip()
    if not sentence:
        return jsonify({"ok": False, "error": "Please enter an English sentence."}), 400

    try:
        bundle = get_model_bundle()
        phrasebook_match = best_phrasebook_match(sentence, bundle["phrasebook"])
        if phrasebook_match:
            return jsonify(
                {
                    "ok": True,
                    "translation": phrasebook_match["translation"],
                    "source_lang": bundle["config"].source_lang,
                    "target_lang": bundle["config"].target_lang,
                    "mode": phrasebook_match["mode"],
                    "matched_source": phrasebook_match["matched_source"],
                    "score": phrasebook_match["score"],
                }
            )

        if bundle["is_demo_dataset"]:
            return jsonify(
                {
                    "ok": False,
                    "error": "This demo model only supports the trained sample phrases. Try one of the listed examples or retrain with a larger dataset.",
                    "mode": "unsupported",
                    "source_lang": bundle["config"].source_lang,
                    "target_lang": bundle["config"].target_lang,
                }
            ), 400

        translated_text = translate_sentence(
            bundle["model"],
            bundle["source_vocab"],
            bundle["target_vocab"],
            bundle["device"],
            sentence,
        )
        return jsonify(
            {
                "ok": True,
                "translation": translated_text or "(no translation generated)",
                "source_lang": bundle["config"].source_lang,
                "target_lang": bundle["config"].target_lang,
                "mode": "model",
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True)
