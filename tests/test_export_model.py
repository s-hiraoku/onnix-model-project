import sys
from pathlib import Path
import pytest
from src.onnx_model_project.export_model import export_onnx_model

# プロジェクトのルートディレクトリをモジュール検索パスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from onnx_model_project.export_model import export_onnx_model

@pytest.fixture
def test_output_dir() -> Path:
    """テスト用の出力ディレクトリを作成"""
    output_dir = Path("onnx_model")
    output_dir.mkdir(exist_ok=True)
    yield output_dir
    # テスト終了後にディレクトリをクリーンアップ
    for file in output_dir.iterdir():
        file.unlink()
    output_dir.rmdir()

def test_export_onnx_model(test_output_dir: Path) -> None:
    """
    モデルエクスポートが正しく行われることを確認。
    """
    model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
    onnx_path = test_output_dir / "test_model.onnx"
    
    # モデルエクスポート
    export_onnx_model(model_name, onnx_path)
    
    # エクスポート後にONNXファイルが存在することを確認
    assert onnx_path.exists(), "ONNXモデルがエクスポートされていません"