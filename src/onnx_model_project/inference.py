import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
from pathlib import Path

def run_inference(onnx_path: Path, question: str, context: str, model_name: str, max_length: int = 512) -> str:
    """
    ONNXモデルを使用して質問応答を実行。

    Args:
        onnx_path (Path): エクスポートされたONNXモデルのパス。
        question (str): 質問。
        context (str): 質問のためのコンテキスト。
        model_name (str): トークナイザーのモデル名。
        max_length (int): 最大入力トークン長。

    Returns:
        str: モデルが予測した回答。
    """
    # ONNXモデルのロード
    session = ort.InferenceSession(onnx_path.as_posix())

    # トークナイザーのロード
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 入力をトークナイズ
    inputs = tokenizer(
        question, context,
        return_tensors="np",  # NumPy形式
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    # モデル入力
    input_feed = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }

    # 推論実行
    outputs = session.run(None, input_feed)
    start_logits, end_logits = outputs
    start_index = start_logits.argmax(axis=1)[0]
    end_index = end_logits.argmax(axis=1)[0]

    # 回答をデコード
    answer = tokenizer.decode(inputs["input_ids"][0][start_index:end_index + 1])
    return answer