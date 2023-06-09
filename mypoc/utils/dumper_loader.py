# 標準
import os
from pathlib import Path
from typing import Any

# サードパーティ
import joblib
import pandas as pd

# ローカル
from mypoc.config.config import config


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
        os.makedirs(os.path.dirname(path), exist_ok=True)
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
        仕様: csv 形式か tsv 形式か pkl 形式のファイルをロードして、データフレームを返す

        Parameters
        ----------
        path: Path
            データを読み込むパス

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
            df = pd.read_csv(path)
        elif ext == ".tsv":
            df = pd.read_csv(path, sep="\t")

        # 仕様: pkl ファイルで保存
        pkl_path = path.with_suffix(".pkl")
        DumperLoader.dump(df, pkl_path)

        return df

    @staticmethod
    def load_train(run_name: str) -> pd.DataFrame:
        """
        仕様: 学習データをロード

        parameters
        ----------
        run_name: str
            ラン名

        Returns
        -------
        pd.DataFrame
            学習データ
        """
        path = Path(config["PATH_INPUT"]) / run_name / "train.pkl"
        return DumperLoader.load_df(path)

    @staticmethod
    def load_test(run_name: str) -> pd.DataFrame:
        """
        仕様: テストデータをロード

        parameters
        ----------
        run_name: str
            ラン名

        Returns
        -------
        pd.DataFrame
            テストデータ
        """
        # テストデータのロード
        path = Path(config["PATH_INPUT"]) / run_name / "test.pkl"
        return DumperLoader.load_df(path)
