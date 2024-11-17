from pathlib import Path
from src.onnx_model_project.export_model import export_onnx_model
from src.onnx_model_project.inference import run_inference

def test_run_inference() -> None:
    """質問応答の推論テスト"""
    model_name = "sonoisa/question-answering"
    output_dir = Path("onnx_model")
    output_dir.mkdir(exist_ok=True)
    onnx_path = output_dir / "qa_model.onnx"

    # モデルをエクスポート
    export_onnx_model(model_name, onnx_path)

    # 推論を実行
    question = "イベントはいつ開催されますか？"
    context = "イベントは2024年11月15日に開催されます。場所は東京です。"
    answer = run_inference(onnx_path, question, context, model_name)

    assert answer == "2024年11月15日", f"期待される回答と一致しません: {answer}"