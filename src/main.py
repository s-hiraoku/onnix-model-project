from pathlib import Path
from onnx_model_project.export_model import export_onnx_model
from onnx_model_project.inference import run_inference

def main() -> None:
    # モデル情報
    model_name = "ybelkada/japanese-roberta-question-answering"
    output_dir = Path("onnx_model")
    output_dir.mkdir(exist_ok=True)
    onnx_path = output_dir / "qa_model.onnx"

    # モデルをONNX形式でエクスポート
    export_onnx_model(model_name, onnx_path)

    # ユーザーからコンテキストと質問を取得
    print("質問応答システムを開始します。")
    context = input("コンテキストを入力してください（例: イベント情報など）: ")
    question = input("質問を入力してください: ")

    # 推論の実行
    answer = run_inference(onnx_path, question, context, model_name)
    print(f"\n質問: {question}")
    print(f"回答: {answer}")

if __name__ == "__main__":
    main()