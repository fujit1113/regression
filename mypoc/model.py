# 標準ライブラリ
from abc import ABC, abstractmethod
from inspect import Parameter, signature
from typing import Any, Callable, Dict, Mapping, Optional, Union

# サードパーティのライブラリ
import numpy as np
import pandas as pd
import optuna

# ローカル（自身のプロジェクト）のライブラリ


class _Model(ABC):
    """
    機械学習モデルのための抽象クラス

    params: ハイパーパラメータ

    このクラスを派生し、派生クラスでは以下のメソッドを実装してください。
    train, predict, save_model, load_model
    """

    def __init__(
        self,
        task: str,
        model_name: str,
        fixed_params: Dict[str, Any],
        make_loss_func=None,
        make_eval_func=None,
    ) -> None:
        """
        コンストラクタ

        Parameters
        ----------
        params : Dict[str, Any]
            モデルの名前
        """
        self.task = task
        self.model_name = model_name
        self.fixed_params = fixed_params
        self.make_loss_func = make_loss_func
        self.make_eval_func = make_eval_func

    @abstractmethod
    def set_params(
        self,
        params: Dict[str, Any],
    ) -> None:
        """
        指定のパラメータをセット

        Parameters
        ----------
        params : Dict[str, Any]
            モデルに関するパラメータ

        Returns
        -------
        None
        """
        raise NotImplementedError

    @abstractmethod
    def _check_loss_func(params: Mapping[str, Parameter]):
        """
        確認: 定義したカスタム損失関数が妥当

        Parameters
        ----------
        params : Mapping[str, Parameter]
            カスタム損失関数の引数と戻り値情報

        Returns
        -------
        None
        """
        raise NotImplementedError

    @abstractmethod
    def _check_eval_func(params: Mapping[str, Parameter]):
        """
        確認: 定義したカスタム評価関数が妥当

        Parameters
        ----------
        params : Mapping[str, Parameter]
            カスタム評価関数の引数と戻り値情報

        Returns
        -------
        None
        """
        raise NotImplementedError

    @abstractmethod
    def train(
        self,
        tr_x: pd.DataFrame,
        tr_y: pd.Series,
        va_x: Optional[pd.DataFrame] = None,
        va_y: Optional[pd.Series] = None,
    ) -> None:
        """
        モデルを学習し、学習済のモデルを保存する

        Parameters
        ----------
        tr_x : pd.DataFrame
            学習データの特徴量

        tr_y : pd.Series
            学習データの目的変数

        va_x : Optional[pd.DataFrame]
            バリデーションデータの特徴量

        va_y : Optional[pd.Series]
            バリデーションデータの目的変数

        Returns
        ----------
        None
        """
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        te_x: pd.DataFrame,
    ) -> np.array:
        """学習済のモデルで予測する

        :param te_x: バリデーションデータやテストデータの特徴量
        :return: 予測値
        """
        raise NotImplementedError

    @abstractmethod
    def save_model(self) -> None:
        """モデルを保存する"""
        raise NotImplementedError

    @abstractmethod
    def load_model(self) -> None:
        """モデルを読み込む"""
        raise NotImplementedError

    @property
    def params_space(cls):
        attrs = "\n".join(f"{k}: {v}" for k, v in cls._params_space.items())
        print(attrs)
