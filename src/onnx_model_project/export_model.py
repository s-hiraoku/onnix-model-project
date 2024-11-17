import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import onnx
import onnxruntime as ort
from typing import Union

def export_onnx_model(
    model_name: str,
    output_path: Union[Path, str],
    max_length: int = 512,
    opset_version: int = 14,
) -> None:
    """
    質問応答タスク用のモデルをONNX形式にエクスポートする関数。

    Args:
        model_name (str): 事前学習済みモデルの名前。
        output_path (Union[Path, str]): ONNX形式で保存するパス。
        max_length (int): 最大入力トークン長。
        opset_version (int): ONNXのopsetバージョン。
    """
    # 保存パスの準備
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # トークナイザーとモデルをロード
    print("モデルとトークナイザーをロード中...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    # サンプル入力の準備（質問＋コンテキスト）
    question = "イベントはいつ開催されますか？"
    context = "今日は2024年10月1日です。イベントは約1ヶ月後の11月15日に開催されます。場所は東京です。"
    inputs = tokenizer(
        question, context,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    # モデルをONNX形式にエクスポート
    print("ONNX形式にエクスポート中...")
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        output_path.as_posix(),
        opset_version=opset_version,
        input_names=["input_ids", "attention_mask"],
        output_names=["start_logits", "end_logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        },
    )
    print(f"ONNXモデルが生成されました: {output_path}")

    # モデルの検証
    verify_onnx_model(output_path)

    # ONNX Runtimeでの推論テスト
    test_onnx_model(output_path, inputs)


def verify_onnx_model(onnx_path: Path) -> None:
    """
    ONNXモデルが正しい形式で生成されているかを検証する。

    Args:
        onnx_path (Path): ONNXモデルのパス。
    """
    print("ONNXモデルを検証中...")
    model = onnx.load(onnx_path.as_posix())
    onnx.checker.check_model(model)
    print("ONNXモデルの検証に成功しました。")


def test_onnx_model(onnx_path: Path, inputs: dict) -> None:
    """
    ONNX Runtimeを使用してモデルの推論をテストする。

    Args:
        onnx_path (Path): ONNXモデルのパス。
        inputs (dict): トークナイザーの出力（input_idsとattention_mask）。
    """
    print("ONNX Runtimeで推論をテスト中...")
    session = ort.InferenceSession(onnx_path.as_posix())

    # 推論を実行
    outputs = session.run(
        None,
        {
            "input_ids": inputs["input_ids"].numpy(),
            "attention_mask": inputs["attention_mask"].numpy(),
        },
    )

    start_logits, end_logits = outputs
    print("ONNXモデルの推論結果:")
    print("  Start logits:", start_logits)
    print("  End logits:", end_logits)


# 実行例
if __name__ == "__main__":
    model_name = "distilbert-base-uncased-distilled-squad"  # 任意の事前学習済みモデル名
    output_path = "./onnx_models/qa_model.onnx"  # 保存先のパス
    export_onnx_model(model_name, output_path)