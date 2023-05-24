import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class _Model(ABC):
    """機械学習モデルのための抽象クラス

    :param run_fold_name: ランの名前とfoldの番号を組み合わせた名前
    :param fixed_hyperparams: 固定するハイパーパラメータ

    このクラスを派生し、派生クラスでは以下のメソッドを実装してください。
    train, predict, save_model, load_model
    """

    def __init__(self, run_fold_name: str, fixed_hyperparams: dict) -> None:
        """コンストラクタ

        :param run_fold_name: ランの名前とfoldの番号を組み合わせた名前
        :param fixed_hyperparams: 固定するハイパーパラメータ
        """
        self.run_fold_name = run_fold_name
        self.fixed_hyperparams = fixed_hyperparams
        self.model = None

    @abstractmethod
    def train(
        self,
        tr_x: pd.DataFrame,
        tr_y: pd.Series,
        va_x: Optional[pd.DataFrame] = None,
        va_y: Optional[pd.Series] = None,
        **kwargs,
    ) -> None:
        """モデルを学習し、学習済のモデルを保存する

        :param tr_x: 学習データの特徴量
        :param tr_y: 学習データの目的変数
        :param va_x: バリデーションデータの特徴量
        :param va_y: バリデーションデータの目的変数
        :param categorical_cols: (CatBoost) カテゴリカル変数のリスト
        """
        pass

    @abstractmethod
    def predict(self, te_x: pd.DataFrame) -> np.array:
        """学習済のモデルで予測する

        :param te_x: バリデーションデータやテストデータの特徴量
        :return: 予測値
        """
        pass

    @abstractmethod
    def save_model(self) -> None:
        """モデルを保存する"""
        pass

    @abstractmethod
    def load_model(self) -> None:
        """モデルを読み込む"""
        pass
