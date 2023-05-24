import os

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna.integration.lightgbm as lgb_optuna

from .model import _Model
from .util import Util


class ModelLGB(_Model):
    def train(self, tr_x, tr_y, va_x=None, va_y=None):
        # データのセット
        validation = va_x is not None
        dtrain = lgb.Dataset(tr_x, label=tr_y)
        if validation:
            dvalid = lgb.Dataset(va_x, label=va_y)

        # ハイパーパラメータチューニング
        tuner = lgb_optuna.LightGBMTuner(
            self.fixed_hyperparams, dtrain, valid_sets=dvalid if validation else None
        )
        tuner.run()

        # 最適なハイパーパラメータで再学習
        best_params = tuner.best_params
        self.model = lgb.train(best_params, dtrain)

    def predict(self, te_x):
        return self.model.predict(te_x)

    def save_model(self):
        model_path = os.path.join("../model/model", f"{self.run_fold_name}.model")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        Util.dump(self.model, model_path)

    def load_model(self):
        model_path = os.path.join("../model/model", f"{self.run_fold_name}.model")
        self.model = Util.load(model_path)
