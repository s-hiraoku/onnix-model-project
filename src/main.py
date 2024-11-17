from pathlib import Path
from onnx_model_project.export_model import export_onnx_model
from onnx_model_project.inference import run_inference
from onnx_model_project.utils import ensure_output_dir

if __name__ == "__main__":
    # モデルエクスポート
    model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
    output_dir = Path("onnx_model")
    ensure_output_dir(output_dir)

    onnx_path = output_dir / "model.onnx"
    export_onnx_model(model_name, onnx_path)

    # 推論の実行
    test_text = "明日は2024年10月1日です。イベントは11月15日に開催されます。"
    run_inference(onnx_path, test_text, model_name)