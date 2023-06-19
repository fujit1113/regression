# 標準ライブラリ
from typing import Any, Callable, Dict, Optional, Tuple

# サードパーティのライブラリ
from catboost import CatBoost, Pool
import numpy as np
from lightgbm import callback
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss
from scipy.special import huber

# ローカルのライブラリ
from .model import _Model
from CPoC.config.config import config
from CPoC.utils.checker import Checker


class ModelCAT(_Model):
    _TUNABLE_PARAMS = config["TUNABLE_PARAMS_CAT"]
    _UNTUNABLE_PARAMS = config["UNTUNABLE_PARAMS_CAT"]

    def init_setting(self):
        """
        仕様: デフォルトの損失関数と評価関数を設定
        """
        # 仕様: デフォルト損失関数を設定
        default_loss_functions = {
            "regression": "RMSE",
            "binary": "Logloss",
            "multiclass": "MultiClass",
        }

        self.loss_function = self.fixed_params.get(
            "loss_function", default_loss_functions[self.task]
        )
        self.eval_metric = self.fixed_params.get("eval_metric", self.loss_function)

    def _separate_params(
        self,
        params: Dict[str, Any],
    ) -> None:
        """
        仕様: 学習時のパラメータの与え方を考慮して、パラメータを分割

        Parameters
        ----------
        params : Dict[str, Any]
            パラメータ
        """
        pass

    def default_loss_func_optuna(self) -> Callable:
        """
        仕様: 目的変数にもとづいて Optuna 用のデフォルト損失関数を設定

        Returns
        -------
        Callable
            Optuna 用のデフォルト損失関数
        """
        if self.objective == "regression":
            return mean_squared_error

        elif self.objective == "regression_l1":
            return mean_absolute_error

        elif self.objective == "huber":

            def huber_loss_error(preds, y):
                return np.mean(huber(y - preds))

            return huber_loss_error

        elif self.objective == "binary":

            def binary_loss_error(preds, y):
                preds_proba = 1.0 / (1.0 + np.exp(-preds))
                return log_loss(y, preds_proba)

            return binary_loss_error

        elif self.objective == "multiclass":

            def multiclass_loss_func(preds, y):
                preds_proba = np.exp(preds) / np.sum(
                    np.exp(preds), axis=1, keepdims=True
                )
                return log_loss(y, preds_proba)

            return multiclass_loss_func

        else:
            raise ValueError("Unknown objective.")

    def train(
        self,
        train_x: pd.DataFrame,
        train_y: pd.DataFrame,
        valid_x: Optional[pd.DataFrame] = None,
        valid_y: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        処理: 与えられた訓練データ(と検証データ)を使って学習

        Parameters
        ----------
        train_x : pd.DataFrame
            訓練データの特徴量

        train_y : pd.DataFrame
            訓練データの目的変数

        valid_x : Optional[pd.DataFrame]
            検証データの特徴量

        valid_y : Optional[pd.DataFrame]
            検証データの目的変数
        """
        # コンストラクタ
        self.model = CatBoost(self.params)

        # データのセット
        train = Pool(train_x, label=train_y)

        if valid_x is not None:
            valid = Pool(valid_x, label=valid_y)
            self.model.fit(
                self.params,
                train,
                eval_sets=valid,
                fobj=self.loss_func,
                feval=self.eval_func,
            )

        else:
            self.model.fit(
                self.params,
                train,
                fobj=self.loss_func,
            )

    @staticmethod
    def _check_loss_func(loss_func: Callable[[np.ndarray, cat.Pool], float]) -> None:
        """
        確認: カスタム損失関数が妥当

        Parameters
        ----------
        loss_func: Callable[[np.ndarray, cat.Pool], float]
            カスタム損失関数
        """
        n_args = Checker.count_arguments(loss_func)

        # 確認1: カスタム損失関数は2つの引数を取る
        if n_args == 2:
            # 確認2: 引数 preds は np.ndarray
            Checker.check_argument_type(loss_func, "preds", np.ndarray)

            # 確認3: 引数 train_data は cat.Pool
            Checker.check_argument_type(loss_func, "train_data", cat.Pool)

            # 確認4: 戻り値は Tuple[np.ndarray, np.ndarray]
            Checker.check_return_type(loss_func, Tuple[np.ndarray, np.ndarray])

        else:
            raise ValueError("The custom loss function must have exactly two arguments")

    @staticmethod
    def _check_eval_func(eval_func: Callable[[np.ndarray, cat.Pool], float]) -> None:
        """
        確認: カスタム評価関数が妥当

        Parameters
        ----------
        eval_func : Callable[[np.ndarray, cat.Pool], float]
            カスタム損失関数
        """
        n_args = Checker.count_arguments(eval_func)

        # 確認1: カスタム損失関数は2つの引数を取る
        if n_args == 2:
            # 確認2: 引数 preds は np.ndarray
            Checker.check_argument_type(eval_func, "preds", np.ndarray)

            # 確認3: 引数 data は cat.Pool
            Checker.check_argument_type(eval_func, "data", cat.Pool)

            # 確認4: 戻り値は Tuple[str, float, bool]
            Checker.check_return_type(eval_func, Tuple[str, float, bool])

        else:
            raise ValueError("The custom eval function must have exactly two arguments")

    def predict(self, te_x):
        return self.model.predict(te_x)
