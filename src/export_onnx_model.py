import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

# モデルの名前と保存先を設定
model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
output_dir = Path("onnx_model")
output_dir.mkdir(exist_ok=True)

# モデルとトークナイザーをロード
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# テキストサンプルをトークナイズ
example_text = "イベントは2024年11月15日に開催されます。"
inputs = tokenizer(example_text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)

# ONNX形式に変換
onnx_path = output_dir / "model.onnx"
torch.onnx.export(
    model,                                # モデル
    (inputs["input_ids"], inputs["attention_mask"]),  # モデルへの入力
    onnx_path.as_posix(),                # 保存先
    opset_version=14,                    # ← opset_versionを14以上に設定
    input_names=["input_ids", "attention_mask"],  # 入力の名前
    output_names=["output"],             # 出力の名前
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
    },  # 動的なシーケンス長をサポート
)

print(f"ONNXモデルが生成されました: {onnx_path}")