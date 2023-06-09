import json
import os
from pathlib import Path

# パス管理
package_dir = Path(__file__).resolve().parent
root_dir = package_dir.parent
config_path = package_dir / "config.json"

# JSON ファイルから設定を読み込む
with open(config_path, "r") as f:
    config = json.load(f)
