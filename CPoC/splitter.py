# 標準ライブラリ
from typing import Tuple, Union
from pathlib import Path

# サードパーティのライブラリ
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

# ローカルのライブラリ
from CPoC.config.config import config
from CPoC.utils.checker import Checker
from CPoC.utils.dumper_loader import DumperLoader
from .validator import Validator


class Splitter:
    @staticmethod
    def split_data(
        path: str,
        bunch_name: str,
        task: str,
        target: str = config["TARGET"],
        test_size: float = config["TEST_SIZE"],
        random_seed: int = config["RANDOM_SEED"],
    ) -> None:
        """
        仕様: データを学習データとテストデータに分ける

        Parameters
        ----------
        path : Path
            データファイルのパス

        bunch_name : str
            バンチの名前

        task : str
            タスクの種類 ('regression' か 'binary' か 'multiclass')

        target : str
            目的変数のカラム名

        test_size : float
            テストデータの割合

        random_seed : int
            データ分割のランダムシード
        """
        path_input = Path(config["PATH_INPUT"]) / bunch_name
        path_learn = path_input / "learn.pkl"
        path_test = path_input / "test.pkl"

        # 確認1: インプットフォルダに学習データがない
        Checker.check_path_no_existence(path_learn)

        # 確認2: インプットフォルダにテストデータがない
        Checker.check_path_no_existence(path_test)

        # 確認3: task が妥当
        Validator.check_task(task)

        # 処理: データをロード
        df = DumperLoader.load_df(path)

        # 確認4: df のカラムは target を含む
        Checker.check_element_inclusion(target, df.columns, f"{target}", "data")

        # 処理: 学習データとテストデータに分ける
        learn, test = Splitter._split_data(df, task, target, test_size, random_seed)

        # 仕様: 保存
        DumperLoader.dump(learn, path_learn)
        DumperLoader.dump(test, path_test)

    @staticmethod
    def _split_data(
        df: pd.DataFrame,
        task: str,
        target: str = config["TARGET"],
        test_size: float = config["TEST_SIZE"],
        random_seed: int = config["RANDOM_SEED"],
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

        random_seed : int
            データ分割のランダムシード

        Returns
        ----------
        Tuple[pd.DataFrame, pd.DataFrame]
            学習データのデータフレーム、テストデータのデータフレーム
        """
        # 処理: 学習データ・テストデータに分ける
        all_x = df.drop(target, axis=1)
        all_y = df[target]

        if task == "regression":
            learn_x, test_x, learn_y, test_y = train_test_split(
                all_x,
                all_y,
                test_size=test_size,
                random_seed=random_seed,
            )
        elif task in ["binary", "multiclass"]:
            learn_x, test_x, learn_y, test_y = train_test_split(
                all_x,
                all_y,
                test_size=test_size,
                stratify=all_y,
                random_seed=random_seed,
            )

        # 仕様: インデックスで結合
        return learn_x.join(learn_y), test_x.join(test_y)

    @staticmethod
    def split_data_time_based(
        path: Path,
        bunch_name: str,
        timestamp_col: str,
        test_size: float = config["TEST_SIZE"],
    ) -> None:
        """
        仕様: データをテストデータと学習データに分ける

        Parameters
        ----------
        path : Path
            データファイルのパス

        bunch_name : str
            バンチの名前

        timestamp_col : str
            タイムスタンプのカラム名

        test_size : float
            テストデータの割合
        """
        path_input = Path(config["PATH_INPUT"]) / bunch_name
        path_learn = path_input / "learn.pkl"
        path_test = path_input / "test.pkl"

        # 確認1: インプットフォルダに学習データがない
        Checker.check_path_no_existence(path_learn)

        # 確認2: インプットフォルダにテストデータがない
        Checker.check_path_no_existence(path_test)

        # 処理: データをロード
        df = DumperLoader.load_df(path)

        # 確認3: df のカラムは timestamp_col を含む
        Checker.check_list_inclusion([timestamp_col], "your input", df.columns, "data")

        # 処理: 学習データとテストデータに分ける
        learn, test = Splitter._split_data_time_based(df, timestamp_col, test_size)

        # 仕様: 保存
        DumperLoader.dump(learn, path_learn)
        DumperLoader.dump(test, path_test)

    @staticmethod
    def _split_data_time_based(
        df: pd.DataFrame,
        timestamp_col: str,
        test_size: float = config["TEST_SIZE"],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        仕様: ある時刻を境に、データを学習データとテストデータに分ける

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
            学習データのデータフレーム、テストデータのデータフレーム
        """
        # 処理: タイムスタンプでソート
        df = df.sort_values(timestamp_col)

        # 処理: 学習データとテストデータに分割
        split_point = int(len(df) * (1 - test_size))

        return df.iloc[:split_point], df.iloc[split_point:]

    @staticmethod
    def k_fold(
        task: str,
        n_fold: int = config["N_FOLD"],
        random_seed: int = config["RANDOM_SEED"],
    ) -> Union[StratifiedKFold, KFold]:
        """
        指定したタスクに適した交差検証オブジェクトを作成

        Parameters
        ----------
        task : str
            タスクの種類 ('regression' か 'binary' か 'multiclass')

        n_fold : int
            フォールドの数

        random_seed : int
            交差検証の乱数シード

        Returns
        ----------
        Union[StratifiedKFold, KFold]
            交差検証オブジェクト
        """
        params = {
            "n_splits": n_fold,
            "shuffle": True,
            "random_seed": random_seed,
        }

        # タスクごとの交差検証
        if task == "regression":
            cv = KFold(**params)
        elif task in ["binary", "multiclass"]:
            cv = StratifiedKFold(**params)
        else:
            raise ValueError(f"Unknown task type: {task}")

        return cv
