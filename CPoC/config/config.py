import yaml
from pathlib import Path

# パス管理
package_dir = Path(__file__).resolve().parent
root_dir = package_dir.parent
config_path = package_dir / "config.yaml"

# YAML ファイルから設定を読み込む
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
