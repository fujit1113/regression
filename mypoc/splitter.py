# 標準ライブラリ
import os
from typing import Tuple, Union
from pathlib import Path

# サードパーティのライブラリ
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

# ローカルのライブラリ / アプリケーション
from mypoc.config.config import config
from mypoc.utils.checker import Checker
from mypoc.utils.dumper_loader import DumperLoader
from .validator import Validator


class Splitter:
    @staticmethod
    def split_data(
        path: str,
        run_name: str,
        task: str,
        target: str = config["TARGET"],
        test_size: float = config["TEST_SIZE"],
        random_state: int = config["RANDOM_STATE"],
    ) -> None:
        """
        仕様: データをテストデータと学習データに分ける

        Parameters
        ----------
        path : Path
            データファイルのパス

        run_name : str
            ランの名前

        task : str
            タスクの種類 ('regression' か 'binary' か 'multiclass')

        target : str
            目的変数のカラム名

        test_size : float
            テストデータの割合

        random_state : int
            データ分割のランダムシード

        Returns
        ----------
        None
        """
        path_input = Path(config["PATH_INPUT"]) / run_name
        path_train = path_input / "train.pkl"
        path_test = path_input / "test.pkl"

        # 確認1: インプットフォルダに学習データがない
        Checker.check_no_path_exists(path_train)

        # 確認2: インプットフォルダにテストデータがない
        Checker.check_no_path_exists(path_test)

        # 確認3: task が妥当
        Validator.check_task(task)

        # 処理: データをロード
        df = DumperLoader.load_df(path)

        # 確認4: df のカラムは target を含む
        Checker.check_list_inclusion([target], "your input", list(df.columns), "data")

        # 処理: 学習データとテストデータに分ける
        train, test = Splitter._split_data(df, task, target, test_size, random_state)

        # 仕様: 保存
        DumperLoader.dump(train, path_train)
        DumperLoader.dump(test, path_test)

    @staticmethod
    def _split_data(
        df: pd.DataFrame,
        task: str,
        target: str = config["TARGET"],
        test_size: float = config["TEST_SIZE"],
        random_state: int = config["RANDOM_STATE"],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        データをテストデータと学習データに分ける
        ※ コンペではなく、案件やオフラインテストで有効

        Parameters
        ----------
        df : pd.DataFrame
            データフレーム

        task : str
            タスクの種類 ('regression' か 'binary' か 'multiclass')

        target : str
            目的変数のカラム名

        test_size : float
            テストデータの割合

        random_state : int
            データ分割のランダムシード

        Returns
        ----------
        Tuple[pd.DataFrame, pd.DataFrame]
            学習データのデータフレーム、テストデータのデータフレーム
        """
        # 処理: 学習データ・テストデータに分ける
        X = df.drop(target, axis=1)
        y = df[target]

        if task == "regression":
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
            )
        elif task in ["binary", "multiclass"]:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                stratify=y,
                random_state=random_state,
            )

        # 仕様: インデックスで結合
        return X_train.join(y_train), X_test.join(y_test)

    @staticmethod
    def split_data_time_based(
        path: Path,
        run_name: str,
        timestamp_col: str,
        test_size: float = config["TEST_SIZE"],
    ) -> None:
        """
        仕様: データをテストデータと学習データに分ける

        Parameters
        ----------
        path : Path
            データファイルのパス

        run_name : str
            ランの名前

        timestamp_col : str
            タイムスタンプのカラム名

        test_size : float
            テストデータの割合
        """
        path_input = Path(config["PATH_INPUT"]) / run_name
        path_train = path_input / "train.pkl"
        path_test = path_input / "test.pkl"

        # 確認1: インプットフォルダに学習データがない
        Checker.check_no_path_exists(path_train)

        # 確認2: インプットフォルダにテストデータがない
        Checker.check_no_path_exists(path_test)

        # 処理: データをロード
        df = DumperLoader.load_df(path)

        # 確認3: df のカラムは timestamp_col を含む
        Checker.check_list_inclusion([timestamp_col], "your input", df.columns, "data")

        # 処理: 学習データとテストデータに分ける
        train, test = Splitter._split_data_time_based(df, timestamp_col, test_size)

        # 仕様: 保存
        DumperLoader.dump(train, path_train)
        DumperLoader.dump(test, path_test)

    @staticmethod
    def _split_data_time_based(
        df: pd.DataFrame,
        timestamp_col: str,
        test_size: float = config["TEST_SIZE"],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        仕様: ある時刻を境に、データをテストデータと学習データに分ける

        Parameters
        ----------
        df : pd.DataFrame
            データフレーム

        timestamp_col : str
            タイムスタンプのカラム

        target : str
            目的変数

        test_size : float
            テストデータの割合

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            訓練データのデータフレーム、テストデータのデータフレーム
        """
        # 処理: タイムスタンプでソート
        df = df.sort_values(timestamp_col)

        # 処理: 訓練データとテストデータに分割
        split_point = int(len(df) * (1 - test_size))

        return df.iloc[:split_point], df.iloc[split_point:]

    @staticmethod
    def k_fold(
        task: str,
        n_fold: int = config["N_FOLD"],
        random_state: int = config["RANDOM_STATE"],
    ) -> Union[StratifiedKFold, KFold]:
        """
        指定したタスクに適した交差検証オブジェクトを作成

        Parameters
        ----------
        task : str
            タスクの種類 ('regression' か 'binary' か 'multiclass')

        n_fold : int
            フォールドの数

        random_state : int
            クロスバリデーションの乱数シード

        Returns
        ----------
        Union[StratifiedKFold, KFold]
            交差検証オブジェクト
        """
        params = {
            "n_splits": n_fold,
            "shuffle": True,
            "random_state": random_state,
        }

        # タスクごとの交差検証
        if task == "regression":
            cv = KFold(**params)
        elif task in ["binary", "multiclass"]:
            cv = StratifiedKFold(**params)
        else:
            raise ValueError(f"Unknown task type: {task}")

        return cv
