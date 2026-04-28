from src.inference import predictor_base, predictor_bert
from src.config_bert import PathConfig, ModelConfig
from huggingface_hub import snapshot_download


def compare_models():
    # paths = PathConfig()
    # model = ModelConfig()
    # modelpath = paths.SAVED_MODELS_PATH / f"bert-{model.version}/checkpoint-20000"
    # bert = predictor_bert.Predictor(modelpath)

    bert_path = snapshot_download(repo_id="sweetguma/bert-sentiment-model")
    bert = predictor_bert.Predictor(bert_path)

    base_model = predictor_base.load_model()

    texts = [
        "I love this product!",
        "This is the worst experience ever.",
        "It's okay, not great but not bad.",
        "I feel so happy today!",
        "I'm really disappointed and sad.",
    ]

    print("=" * 60)
    print(" BASELINE vs BERT COMPARISON")
    print("=" * 60)

    for text in texts:
        bert_result = bert.predict(text)
        base_result = predictor_base.predict(base_model, text)

        print(f"\n📝 Text: {text}")
        print("-" * 60)

        print(
            f"Baseline → pred: {base_result['prediction']} | conf: {base_result.get('confidence', 'N/A'):.4f}"
        )
        print(
            f"BERT     → pred: {bert_result['prediction']} | conf: {bert_result.get('confidence', 'N/A'):.4f}"
        )

        # disagreement
        if base_result["prediction"] != bert_result["prediction"]:
            print("⚠️  DISAGREEMENT!")

    print("\nDone.")


if __name__ == "__main__":
    compare_models()
