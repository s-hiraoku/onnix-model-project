import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

def export_onnx_model(model_name: str, output_path: Path, max_length: int = 128) -> None:
    """
    モデルをONNX形式にエクスポートする関数。

    Args:
        model_name (str): 事前学習済みモデルの名前。
        output_path (Path): ONNX形式で保存するパス。
        max_length (int): 最大入力トークン長。
    """
    # トークナイザーとモデルのロード
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # サンプル入力を準備
    sample_text = "サンプルテキストです。"
    inputs = tokenizer(
        sample_text,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    # モデルをONNX形式にエクスポート
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        output_path.as_posix(),
        opset_version=14,
        input_names=["input_ids", "attention_mask"],
        output_names=["output"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        },
    )
    print(f"モデルがエクスポートされました: {output_path}")