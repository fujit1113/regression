# 標準ライブラリ
from pathlib import Path
from typing import Any

# サードパーティのライブラリ
import joblib
import pandas as pd

# ローカルのライブラリ
from CPoC.config.config import config


class DumperLoader:
    """
    仕様: データの保存と読み込み
    """

    @staticmethod
    def dump(
        value: Any,
        path: Path,
    ) -> None:
        """
        仕様: 与えられたパスに値を保存

        Parameters
        ----------
        value : object
            保存するオブジェクト

        path : Path
            オブジェクトを保存するパス
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(value, path, compress=True)

    @staticmethod
    def load(path: Path) -> Any:
        """
        仕様: 与えられたパスからデータをロード

        Parameters
        ----------
        path : Path
            データを読み込むパス

        Returns
        -------
        object
            読み込まれたオブジェクト
        """
        return joblib.load(path)

    @staticmethod
    def load_df(path: Path) -> pd.DataFrame:
        """
        仕様: csv 形式か tsv 形式か pkl 形式のファイルをロードして、\n
        データフレームを返す

        Parameters
        ----------
        path: Path
            ファイルパス

        Returns
        ----------
        pd.DataFrame
        """
        # 確認1: ファイルが csv か TSV か pkl 形式
        ext = path.suffix
        if ext not in [".csv", ".tsv", ".pkl"]:
            raise ValueError("The file extension must be either .csv, .tsv or .pkl")

        # 処理: ファイルをロード
        if ext == ".pkl":
            df = DumperLoader.load(path)
        elif ext == ".csv":
            df = DumperLoader._load_df_and_dump(path)
        elif ext == ".tsv":
            df = DumperLoader._load_df_and_dump(path, separator="\t")

        return df

    @staticmethod
    def _load_df_and_dump(
        path: Path,
        separator: str = ",",
    ) -> pd.DataFrame:
        """
        処理: ファイルをロードし、データフレームを返し、pkl 形式でデータを保存

        Parameters
        ----------
        path: Path
            ファイルパス

        separator: str
            データのセパレータ（デフォルトは ','）

        Returns
        ----------
        pd.DataFrame
        """
        df = pd.read_csv(path, sep=separator)
        pkl_path = path.with_suffix(".pkl")
        DumperLoader.dump(df, pkl_path)
        return df

    @staticmethod
    def load_learn(bunch_name: str) -> pd.DataFrame:
        """
        仕様: 学習データをロード

        parameters
        ----------
        bunch_name: str
            バンチの名前

        Returns
        -------
        pd.DataFrame
            学習データ
        """
        path = Path(config["PATH_INPUT"]) / bunch_name / "learn.pkl"
        return DumperLoader.load_df(path)

    @staticmethod
    def load_test(bunch_name: str) -> pd.DataFrame:
        """
        仕様: テストデータをロード

        parameters
        ----------
        bunch_name: str
            バンチの名前

        Returns
        -------
        pd.DataFrame
            テストデータ
        """
        # テストデータのロード
        path = Path(config["PATH_INPUT"]) / bunch_name / "test.pkl"
        return DumperLoader.load_df(path)
