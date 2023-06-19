# 標準ライブラリ
from abc import ABC, abstractmethod
from inspect import Parameter, signature
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

# サードパーティのライブラリ
import numpy as np
import optuna
import pandas as pd

# ローカルのライブラリ
from CPoC.config.config import config
from CPoC.utils.dumper_loader import DumperLoader
from CPoC.utils.checker import Checker


class _Model(ABC):
    """
    機械学習モデルのための抽象クラス

    params: ハイパーパラメータ

    このクラスを派生し、派生クラスでは以下のメソッドを実装してください。
    train, predict, save_model, load_model
    """

    def __init__(
        self,
        bunch_name: str,
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
        task: str
            タスクの種類

        bunch_name:

        params : Dict[str, Any]
            モデルの名前
        """
        self.bunch_name = bunch_name
        self.task = task
        self.model_name = model_name
        self.fixed_params = fixed_params
        self.make_loss_func = make_loss_func
        self.make_eval_func = make_eval_func
        self.init_setting()
        self.initialize_funcs()

    @abstractmethod
    def init_setting(self) -> None:
        """
        処理: コンストラクタ直後に実行したいこと
        """
        raise NotImplementedError

    def initialize_funcs(self) -> None:
        """
        処理: カスタム損失関数とカスタム評価関数を初期化
        """
        # 処理1: 損失関数を初期化
        self.loss_func = None
        self.loss_func_optuna_valid = None
        self.loss_func_optuna_train = None

        # 処理2: 評価関数を初期化
        self.eval_func = None

    def suggest_params(
        self,
        trial: optuna.trial.Trial,
    ):
        """
        処理: 各ハイパーパラメータの探索範囲を設定し、Optuna にパラメータを提案させる。\n
        仕様: 固定のハイパーパラメータは探索対象外、その値をそのまま使用。

        Parameters
        ----------
        trial : optuna.trial.Trial
            Optunaの trial インスタンス
        """
        params = {}

        for param, space in self._TUNABLE_PARAMS.items():
            if param not in self.fixed_params.keys():
                if space["type"] == "float":
                    params[param] = trial.suggest_float(
                        param, space["low"], space["high"]
                    )
                elif space["type"] == "int":
                    params[param] = trial.suggest_int(
                        param, space["low"], space["high"]
                    )
                elif space["type"] == "categorical":
                    params[param] = trial.suggest_categorical(param, space["choices"])
                else:
                    raise ValueError(
                        "Unknown space type for parameter '{}': {}".format(
                            param, space["type"]
                        )
                    )
            else:
                params[param] = self.fixed_params[param]

        # 仕様: モデルの学習の仕方に応じて、ハイパーパラメータの渡し方を適宜変える
        self._separate_params(params)

    def set_params(
        self,
        tuned_params: Dict,
    ) -> None:
        params = {**self.fixed_params, **tuned_params}

        # 仕様: train に応じて、ハイパーパラメータの渡し方を適宜変える
        self._separate_params(params)

    @abstractmethod
    def _separate_params(
        self,
        params: Dict[str, Any],
    ) -> None:
        """
        処理: train に応じて、ハイパーパラメータの渡し方を適宜変える

        Parameters
        ----------
        params: Dict[str, Any]
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
            検証データの特徴量

        va_y : Optional[pd.Series]
            検証データの目的変数

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

    def save_model(
        self,
        model_name: str,
    ) -> None:
        # 確認1: model_name は str
        Checker.check_param_type(model_name, "model_name", str)

        # 確認2: model_name は空文字ではない
        Checker.check_existence(model_name, "model_name")

        # 処理: モデルを保存
        path = Path(config["PATH_MODEL"]) / f"{self.bunch_name}" / f"{model_name}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        DumperLoader.dump(self.model, path)

    @staticmethod
    def load_model(
        bunch_name: str,
        model_name: str,
    ):
        # 確認1: bunch_name は str
        Checker.check_param_type(bunch_name, "bunch_name", str)

        # 確認2: bunch_name は空文字ではない
        Checker.check_existence(bunch_name, "bunch_name")

        # 確認3: model_name は str
        Checker.check_param_type(model_name, "model_name", str)

        # 確認4: model_name は空文字ではない
        Checker.check_existence(model_name, "model_name")

        # 確認5: バンチのモデルフォルダにモデルファイルがある
        path = Path(config["PATH_MODEL"]) / f"{bunch_name}" / f"{model_name}.pkl"
        Checker.check_path_existence(path)

        return DumperLoader.load(path)

    @property
    def tunable_params(cls):
        attrs = "\n".join(f"{k}: {v}" for k, v in cls._TUNABLE_PARAMS.items())
        print(attrs)

    @property
    def untunable_params(cls):
        attrs = "\n".join(f"{k}: {v}" for k, v in cls._UNTUNABLE_PARAMS.items())
        print(attrs)
