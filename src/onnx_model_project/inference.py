import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
from pathlib import Path

def run_inference(onnx_path: Path, test_text: str, model_name: str, max_length: int = 128) -> None:
    """
    ONNXモデルを使用してテキストの推論を実行する関数。

    Args:
        onnx_path (Path): エクスポートされたONNXモデルのパス。
        test_text (str): 推論を行う入力テキスト。
        model_name (str): トークナイザーのロードに使用するモデル名。
        max_length (int): 最大入力トークン長。
    """
    # ONNXモデルのロード
    session = ort.InferenceSession(onnx_path.as_posix())

    # トークナイザーのロード
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 入力テキストをトークナイズ
    inputs = tokenizer(
        test_text,
        return_tensors="np",  # NumPy形式
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    # ONNXモデル用の入力を準備
    input_feed = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }

    # 推論を実行
    outputs = session.run(None, input_feed)
    logits = outputs[0]
    predicted_label = np.argmax(logits, axis=1)
    print(f"入力: {test_text}")
    print(f"予測ラベル: {predicted_label}")