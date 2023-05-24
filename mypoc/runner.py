import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold
from typing import Callable, List, Optional, Tuple, Union
from tqdm import tqdm

from .model import _Model
from .util import Logger, Util

logger = Logger()
_RANDOM_STATE: int = 3655


class Runner:
    def __init__(
        self,
        run_name: str,
        model_cls: Callable[[str, dict], _Model],
        features: List[str],
        params: dict,
        task: str = "classification",
        n_fold: int = 5,
    ):
        """コンストラクタ

        :param run_name: ランの名前
        :param model_cls: モデルのクラス
        :param features: 特徴量のリスト
        :param params: ハイパーパラメータ
        :param task: タスクの種類 ('classification' または 'regression')
        :param n_fold: 交差検証のフォールド数
        """
        self.run_name = run_name
        self.model_cls = model_cls
        self.features = features
        self.params = params
        self.task = task
        self.n_fold = n_fold

    def train_fold(
        self, i_fold: Union[int, str]
    ) -> Tuple[_Model, Optional[np.array], Optional[np.array], Optional[float]]:
        """クロスバリデーションでのfoldを指定して学習・評価する

        他のメソッドから呼び出すほか、単体でも確認やパラメータ調整に用いる

        :param i_fold: foldの番号（すべてのときには'all'とする）
        :return: （モデルのインスタンス、レコードのインデックス、予測値、評価によるスコア）のタプル
        """
        # 学習データの読込
        validation = i_fold != "all"
        train_x = self.load_x_train()
        train_y = self.load_y_train()

        if validation:
            # i_foldの妥当性をチェック
            if not isinstance(i_fold, int) or i_fold < 0 or i_fold >= self.n_fold:
                raise ValueError(
                    f"i_fold should be a value between 0 and {self.n_fold - 1}. Got {i_fold} instead."
                )

            # 学習データ・バリデーションデータをセットする
            tr_idx, va_idx = self.load_index_fold(i_fold)
            tr_x, tr_y = train_x.iloc[tr_idx], train_y.iloc[tr_idx]
            va_x, va_y = train_x.iloc[va_idx], train_y.iloc[va_idx]

            # 学習
            model = self.build_model(i_fold)
            model.train(tr_x, tr_y, va_x, va_y)

            # バリデーションデータへの予測・評価
            va_pred = model.predict(va_x)

            # タスクによって評価指標を変更
            if self.task == "classification":
                score = log_loss(va_y, va_pred, eps=1e-15, normalize=True)
            elif self.task == "regression":
                score = mean_squared_error(va_y, va_pred)
            else:
                raise ValueError(f"Unknown task type: {self.task}")

            # モデル、インデックス、予測値、評価を返す
            return model, va_idx, va_pred, score
        else:
            # 学習データ全てで学習
            model = self.build_model(i_fold)
            model.train(train_x, train_y)

            # モデルを返す
            return model, None, None, None

    def run_train_cv(self) -> None:
        """クロスバリデーションで学習・評価

        学習・評価とともに、各foldのモデルを保存、スコアをログ出力する
        """
        logger.info(f"{self.run_name} - start training cv")

        scores = []
        va_idxes = []
        preds = []

        # 各foldで学習
        for i_fold in tqdm(range(self.n_fold), desc="Processing folds"):
            # 学習
            logger.info(f"{self.run_name} fold {i_fold} - start training")
            model, va_idx, va_pred, score = self.train_fold(i_fold)
            logger.info(f"{self.run_name} fold {i_fold} - end training - score {score}")

            # モデルを保存
            model.save_model()

            # 結果を保持
            va_idxes.append(va_idx)
            scores.append(score)
            preds.append(va_pred)

        # 各foldの結果をまとめる
        va_idxes = np.concatenate(va_idxes)
        order = np.argsort(va_idxes)
        preds = np.concatenate(preds, axis=0)
        preds = preds[order]

        logger.info(f"{self.run_name} - end training cv - score {np.mean(scores)}")

        # 予測結果の保存
        Util.dump(preds, f"../model/pred/{self.run_name}-train.pkl")

        # 評価結果の保存
        logger.result_scores(self.run_name, scores)

    def run_predict_cv(self) -> None:
        """クロスバリデーションで学習した各foldのモデルの平均により、テストデータを予測する

        あらかじめrun_train_cvを実行しておく必要がある
        """
        logger.info(f"{self.run_name} - start prediction cv")

        test_x = self.load_x_test()

        preds = []

        # 各foldのモデルで予測
        for i_fold in tqdm(range(self.n_fold), desc="Processing folds"):
            logger.info(f"{self.run_name} - start prediction fold:{i_fold}")
            model = self.build_model(i_fold)
            model.load_model()
            pred = model.predict(test_x)
            preds.append(pred)
            logger.info(f"{self.run_name} - end prediction fold:{i_fold}")

        # 予測の平均値を出力
        pred_avg = np.mean(preds, axis=0)

        # 予測結果の保存
        Util.dump(pred_avg, f"../model/pred/{self.run_name}-test.pkl")

        logger.info(f"{self.run_name} - end prediction cv")

    def run_train_all(self) -> None:
        """学習データすべてで学習し、そのモデルを保存"""
        logger.info(f"{self.run_name} - start training all")

        # 学習データ全てで学習
        i_fold = "all"
        model, _, _, _ = self.train_fold(i_fold)
        model.save_model()

        logger.info(f"{self.run_name} - end training all")

    def run_predict_all(self) -> None:
        """学習データすべてで学習したモデルでテストデータを予測

        あらかじめrun_train_allを実行しておくこと
        """
        logger.info(f"{self.run_name} - start prediction all")

        test_x = self.load_x_test()

        # 学習データすべてで学習したモデルで予測
        i_fold = "all"
        model = self.build_model(i_fold)
        model.load_model()
        pred = model.predict(test_x)

        # 予測結果の保存
        Util.dump(pred, f"../model/pred/{self.run_name}-test.pkl")

        logger.info(f"{self.run_name} - end prediction all")

    def build_model(self, i_fold: Union[int, str]) -> _Model:
        """クロスバリデーションでのfoldを指定して、モデルを作成

        :param i_fold: foldの番号
        :return: モデルのインスタンス
        """
        # ラン名、fold、モデルのクラスからモデルを作成する
        run_fold_name = f"{self.run_name}-{i_fold}"
        return self.model_cls(run_fold_name, self.params)

    def load_x_train(self) -> pd.DataFrame:
        """学習データの特徴量を読み込む

        :return: 学習データの特徴量
        """
        # 学習データのロード
        df = Runner.load_data("../input/train")

        # 列名で抽出する以上のことを行う場合、このメソッドの修正が必要
        return df[self.features]

    def load_y_train(self) -> pd.Series:
        """学習データの目的変数を読み込む

        :return: 学習データの目的変数
        """
        # 学習データのロード
        df = Runner.load_data("../input/train")

        # 目的変数を読込む
        train_y = df["target"]
        train_y = np.array([int(st[-1]) for st in train_y]) - 1
        train_y = pd.Series(train_y)
        return train_y

    def load_x_test(self) -> pd.DataFrame:
        """テストデータの特徴量を読み込む

        :return: テストデータの特徴量
        """
        # テストデータのロード
        df = Runner.load_data("../input/test")

        # 列名で抽出する以上のことを行う場合、このメソッドの修正が必要
        return df[self.features]

    def load_index_fold(self, i_fold: int) -> np.array:
        """クロスバリデーションでのfoldを指定して対応するレコードのインデックスを返す

        :param i_fold: foldの番号
        :return: foldに対応するレコードのインデックス
        """
        # インデックス保存用のファイルパスを指定
        index_file_path = f"../model/fold_indices.pkl"

        if os.path.exists(index_file_path):
            # すでにインデックスファイルが存在する場合、それを読み込む
            with open(index_file_path, "rb") as f:
                fold_indices = pickle.load(f)

        else:
            # 学習データ・バリデーションデータを分けるインデックスを返す
            train_y = self.load_y_train()
            dummy_x = np.zeros(len(train_y))
            cv = Runner.k_fold(self.task, self.n_fold)
            fold_indices = list(cv.split(dummy_x, train_y))

            # インデックスをファイルとして保存
            with open(index_file_path, "wb") as f:
                pickle.dump(fold_indices, f)

        return fold_indices[i_fold]

    @staticmethod
    def load_data(path: str) -> pd.DataFrame:
        """指定のデータを読み込む

        :param path: データの相対パス (拡張子を除く)
        :return: 指定のデータ
        """
        # データのパス
        pkl_path = path + ".pkl"
        csv_path = path + ".csv"

        # pklファイルが存在するか確認
        if os.path.exists(pkl_path):
            df = pd.read_pickle(pkl_path)
        else:
            # pklが存在しない場合、csvから読み込む
            df = pd.read_csv(csv_path)
            # pklファイルを生成
            df.to_pickle(pkl_path)

        # 指定のデータを返す
        return df

    @staticmethod
    def split_data(
        path: str, task: str, test_size: float = 0.2, random_state: int = _RANDOM_STATE
    ) -> None:
        """データをテストデータと学習データに分ける
        ※ コンペではなく、解析案件(テストデータが与えられていない)の場合

        :param path: データの相対パス (拡張子を除く)
        :param task: タスクの種類。'classification'または'regression'
        :param test_size: テストデータの割合。
        :param random_state: データ分割のランダムシード。
        :return: 学習データとテストデータのタプル
        """
        # インデックス保存用のファイルパスを指定
        df = Runner.load_data(path)

        # 学習データ・テストデータを分けるインデックスを返す
        X = df.drop("target", axis=1)
        y = df["target"]
        cv = Runner.k_fold(task, 1)
        X_train, X_test, y_train, y_test = cv.split(X, y)

        # インデックスで結合
        train = X_train.join(y_train)
        test = X_test.join(y_test)

        # 保存
        train.to_pickle(path + "train.pkl")
        test.to_pickle(path + "test.pkl")

    @staticmethod
    def k_fold(task: str, n_fold: int) -> Union[StratifiedKFold, KFold]:
        """指定されたタスクに基づいて適切な交差検証オブジェクトを作成

        :param task: タスクの種類。"classification"または"regression"
        :param n_fold: 分割するフォールドの数
        :return: タスクに対応する交差検証オブジェクト。タスクが"classification"の場合はStratifiedKFold、"regression"の場合はKFoldオブジェクトを返します。
        :raises ValueError: タスクが "classification" または "regression" 以外の場合。
        """
        # タスクごとの交差検証方法
        if task == "classification":
            cv = StratifiedKFold(
                n_splits=n_fold, shuffle=True, random_state=_RANDOM_STATE
            )
        elif task == "regression":
            cv = KFold(n_splits=n_fold, shuffle=True, random_state=_RANDOM_STATE)
        else:
            raise ValueError(f"Unknown task type: {task}")

        return cv
