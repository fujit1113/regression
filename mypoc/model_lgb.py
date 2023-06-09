# 標準ライブラリ
import os
from inspect import Signature
from typing import Any, Callable, Dict, List, Mapping, Tuple, Union

# サードパーティのライブラリ
import lightgbm as lgb
import numpy as np
import pandas as pd
import optuna.integration.lightgbm as lgb_optuna
import optuna

# ローカル（自身のプロジェクト）のライブラリ
from .model import _Model
from mypoc.config.config import config
from mypoc.utils.dumper_loader import DumperLoader


class ModelLGB(_Model):
    _PARAM_SPACE = config["PARAM_SPACE_LGB"]

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

    def set_params(
        self,
        trial: optuna.trial.Trial,
    ) -> Dict:
        params = {}

        for param, space in self._PARAM_SPACE.items():
            # 仕様: objective を指定していないとき、self.task の値で固定
            if param == "objective":
                params[param] = self.fixed_params.get(param, self.task)

            elif param not in self.fixed_params:
                if space["type"] == "uniform":
                    params[param] = trial.suggest_uniform(
                        param, space["low"], space["high"]
                    )
                elif space["type"] == "loguniform":
                    params[param] = trial.suggest_loguniform(
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

        # 処理: lgb.train に直接渡すパラメータとそれ以外のパラメータを分ける
        self.train_params = {
            k: v for k, v in params.items() if k not in ["early_stopping_rounds"]
        }
        self.control_params = {
            k: v for k, v in params.items() if k in ["early_stopping_rounds"]
        }

    @staticmethod
    def _check_loss_func(sig: Signature):
        """
        確認: 定義したカスタム損失関数が妥当

        Parameters
        ----------
        params : Signature
            カスタム損失関数の引数と戻り値情報

        Returns
        -------
        None
        """
        params = sig.parameters

        # 確認1: カスタム損失関数は2つの引数を取る
        for param in params.values():
            print(f"Parameter: {param.name}, Expected type: {param.annotation}")
            print(f"Expected return type: {sig.return_annotation}")
        if len(params) != 2:
            raise ValueError("The custom loss function must have exactly two arguments")

        # 確認2: 2つの引数は "preds" と "train_data"
        if ("preds" not in params) or ("train_data" not in params):
            raise ValueError("The arguments must be named 'preds' and 'train_data'")

        # 確認3: "preds" のデータ型は np.ndarray
        if params["preds"].annotation != np.ndarray:
            raise ValueError("The data type of 'preds' must be np.ndarray")

        # 確認4: "train_data" のデータ型は lgb.Dataset
        if params["train_data"].annotation != lgb.Dataset:
            raise ValueError("The data type of 'train_data' must be lgb.Dataset")

        # 確認5: 戻り値は2つの np.ndarray からなる Tuple
        if sig.return_annotation != Tuple[np.ndarray, np.ndarray]:
            raise ValueError("The return type must be a tuple of two np.ndarray")

    @staticmethod
    def _check_eval_func(sig: Signature):
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
        params = sig.parameters

        if len(params) != 2:
            raise ValueError("The custom loss function must have exactly two arguments")

        # 確認2-2: 2つの引数は "preds" と "train_data"
        if ("preds" not in params) or ("train_data") not in params:
            raise ValueError("The arguments must be named 'preds' and 'train_data'")

        # 確認2-3: "preds" のデータ型は np.ndarray
        if params["preds"].annotation != np.ndarray:
            raise ValueError("The data type of 'preds' must be np.ndarray")

        # 確認2-4: "train_data" のデータ型は lgb.Dataset
        if params["train_data"].annotation != lgb.Dataset:
            raise ValueError("The data type of 'train_data' must be lgb.Dataset")

        # 確認2-5: 戻り値は str, float, bool からなる Tuple
        if sig.return_annotation != Tuple[str, float, bool]:
            raise ValueError("The return type must be a tuple of (str, float, bool)")

    def train(
        self,
        tr_x,
        tr_y,
        va_x=None,
        va_y=None,
    ):
        # データのセット
        dtrain = lgb.Dataset(tr_x, label=tr_y)

        if va_x is not None:
            dvalid = lgb.Dataset(va_x, label=va_y)

            # 注意: feval は大きい値の方が、望ましい
            self.model = lgb.train(
                self.train_params,
                dtrain,
                valid_sets=dvalid,
                fobj=self.loss_func_tr,
                feval=self.eval_func_va,
                **self.control_params,
            )
        else:
            self.model = lgb.train(self.params, dtrain, fobj=self.loss_func_tr)

    def predict(self, te_x):
        return self.model.predict(te_x)

    def save_model(self):
        model_path = os.path.join(
            "../model/" + self.run_name, f"{self.model_name}.model"
        )
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        DumperLoader.dump(self.model, model_path)

    def load_model(self):
        model_path = os.path.join(
            "../model/" + self.run_name, f"{self.model_name}.model"
        )
        self.model = DumperLoader.load(model_path)
