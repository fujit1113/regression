import os
from typing import List, Optional

import numpy as np
import pandas as pd
import catboost as cat
import optuna
from optuna.integration import OptunaSearchCV

from .model import _Model
from .util import Util

_PARAMS_CATBOOST_TUNED: dict = {
    "iterations": optuna.distributions.IntDistribution(100, 1000),
    "depth": optuna.distributions.IntDistribution(1, 10),
    "learning_rate": optuna.distributions.FloatDistribution(1e-4, 1),
    "l2_leaf_reg": optuna.distributions.FloatDistribution(1e-8, 100),
    "border_count": optuna.distributions.IntDistribution(1, 255),
}


class ModelCat(_Model):
    def __init__(
        self,
        run_name: str,
        fixed_hyperparams: dict,
    ):
        super().__init__(self, run_name, fixed_hyperparams)
        self.model_name += "_cat"

    def train(self, tr_x, tr_y, va_x=None, va_y=None, categorical_cols=None):
        # カテゴリカル変数の選定
        cat_features_indices = self._prepare_categorical_features(
            tr_x, categorical_cols
        )

        # データのセット
        validation = va_x is not None
        dtrain = cat.Pool(data=tr_x, label=tr_y, cat_features=cat_features_indices)
        if validation:
            dvalid = cat.Pool(va_x, label=va_y, cat_features=cat_features_indices)

        # ハイパーパラメータチューニング
        optuna_search = OptunaSearchCV(
            cat.CatBoostRegressor(**self.params_fixed),
            _PARAMS_CATBOOST_TUNED,
            cv=self.kf,
            scoring="neg_root_mean_squared_error",
            random_state=self._random_seed,
            verbose=0,
        )
        if validation:
            optuna_search.fit(tr_x, tr_y, eval_set=[dvalid])
        else:
            optuna_search.fit(tr_x, tr_y)

        # 最適なハイパーパラメータで再学習
        best_params = optuna_search.best_params_
        self.model = cat.CatBoostClassifier(**best_params)
        self.model.fit(dtrain)

    def predict(self, te_x):
        return self.model.predict(te_x)

    def save_model(self):
        model_path = os.path.join("../model/model", f"{self.run_fold_name}.model")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        Util.dump(self.model, model_path)

    def load_model(self):
        model_path = os.path.join("../model/model", f"{self.run_fold_name}.model")
        self.model = Util.load(model_path)

    def _prepare_categorical_features(self, df: pd.DataFrame, categorical_cols=None):
        # 所与のカテゴリカル変数のカラムの有無を確認
        if categorical_cols is not None:
            if not set(categorical_cols).issubset(df.columns):
                missing_cols = set(categorical_cols) - set(df.columns)
                raise ValueError(
                    f"Columns {missing_cols} do not exist in the pd.DataFrame instance"
                )

        # dfのカテゴリカル変数リストを取得
        df_categorical_cols = self.get_categorical_features(df)

        # 所与のカテゴリカル変数が与えられている場合、dfのカテゴリカル変数リストとマージ
        if categorical_cols is not None:
            categorical_cols = list(set(df_categorical_cols + categorical_cols))

        else:
            categorical_cols = df_categorical_cols

        return df.columns.get_indexer_for(categorical_cols)

    @staticmethod
    def get_categorical_features(df: pd.DataFrame) -> Optional[List[str]]:
        categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        return categorical_cols
