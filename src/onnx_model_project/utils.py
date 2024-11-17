from pathlib import Path

def ensure_output_dir(output_dir: Path) -> None:
    """
    出力ディレクトリを作成するヘルパー関数。

    Args:
        output_dir (Path): 出力ディレクトリのパス。
    """
    if not output_dir.exists():
        output_dir.mkdir(parents=True)