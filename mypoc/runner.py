# 標準ライブラリ
import os
import pickle
from typing import Callable, Optional, Tuple, Union

# サードパーティのライブラリ
import numpy as np
import pandas as pd
import optuna
from tqdm import tqdm

# ローカルのライブラリ / アプリケーション
from .config.config import config
from .splitter import Splitter
from .model import _Model
from .validator import Validator
from mypoc.utils.dumper_loader import DumperLoader
from mypoc.utils.logger import Logger

logger = Logger()


class Runner:
    def __init__(self, validator: Validator):
        """
        コンストラクタ

        Parameters
        ----------
        validator : Validator
            条件: アトリビュート is_available が True
        """
        # 確認1: runner_params は妥当
        if validator._is_available:
            self.run_name = validator.run_name
            self.task = validator.task
            self.target = validator.target
            self.additions = validator.additions
            self.features = validator.features
            self.description = validator.description
            self.n_fold = validator.n_fold
            self.models = validator.models

            path_model = os.path.join(config["PATH_MODEL"], self.run_name)
            os.makedirs(path_model, exist_ok=True)

        else:
            raise ValueError(
                "The provided Validator instance is not available. Please make sure the 'is_available' attribute of your Validator instance is set to True."
            )

    def run_train_cv(self) -> pd.DataFrame:
        """
        クロスバリデーションで学習・評価。
        損失関数値と評価指標の値を出力し、各フォールドのモデルを保存する。
        """
        # クロスバリデーション開始
        logger.info(f"{self.run_name} - start training cv")

        # 各フォールドで学習
        for i_fold in tqdm(range(self.n_fold), desc="Processing folds"):
            logger.info(f"{self.run_name} fold {i_fold} - start training")
            self._train_fold(i_fold)
            logger.info(f"{self.run_name} fold {i_fold} - end training")

        # クロスバリデーション終了
        logger.info(f"{self.run_name} - end training cv")

    def _train_fold(
        self,
        i_fold: Union[int, str],
        n_trials: int = config["N_TRIALS"],
    ) -> None:
        """
        クロスバリデーションでフォールドを指定して学習・評価する。

        Parameters
        ----------
        i_fold : Union[int, str]
            フォールドの番号（すべてのときには'all'とする)

        n_trials : int
            ハイパーパラメータの探索回数

        Returns
        -------
        pd.DataFrame
            モデル名、損失関数値、評価指標をカラムにもつデータフレーム
        """
        # データをロード
        validation = i_fold != "all"
        train = DumperLoader.load_train(self.run_name)
        train_x = train[self.features]
        train_y = train[self.target]
        train_z = train[self.additions]

        if validation:
            # 学習データ・バリデーションデータをロード・セット
            tr_idx, va_idx = self.load_index_fold(i_fold)
            tr_x, tr_y, tr_z = (
                train_x.iloc[tr_idx],
                train_y.iloc[tr_idx],
                train_z.iloc[tr_idx],
            )
            va_x, va_y, va_z = (
                train_x.iloc[va_idx],
                train_y.iloc[va_idx],
                train_z.iloc[va_idx],
            )

            self._train_models(i_fold, tr_x, tr_y, tr_z, va_x, va_y, va_z, n_trials)

        # 全学習データで学習
        else:
            self._train_models(i_fold, train_x, train_y, train_z, n_trials)

    def _train_models(
        self,
        i_fold: Union[int, str],
        tr_x: pd.DataFrame,
        tr_y: pd.DataFrame,
        tr_z: pd.DataFrame,
        va_x: Optional[pd.DataFrame] = None,
        va_y: Optional[pd.DataFrame] = None,
        va_z: Optional[pd.DataFrame] = None,
        n_trials: int = config["N_TRIALS"],
        random_seed: int = config["RANDOM_STATE"],
    ) -> None:
        """
        交差検証のフォールドを指定してモデルを作る

        Parameters
        ----------
        i_fold : Union[int, str]
            交差検証のフォールド番号 (全データを使う場合は 'all')

        tr_x : pd.DataFrame
            訓練データの特徴量

        tr_y : pd.DataFrame
            訓練データの目的変数

        tr_z : pd.DataFrame
            訓練データのカスタム損失関数やカスタム評価関数に必要な変数

        va_x : Optional[pd.DataFrame]
            バリデーションデータの特徴量。

        va_y : Optional[pd.DataFrame]
            バリデーションデータの目的変数。

        va_z : Optional[pd.DataFrame]
            バリデーションデータの訓練データのカスタム損失関数やカスタム評価関数に必要な変数

        n_trials : int
            Optuna での試行回数

        random_seed : int
            乱数シード

        Returns
        -------
        Tuple[list[str], list[float], Optional[list[float]], Optional[list[float]]]
            モデル名のリスト、訓練データに対する損失関数値のリスト、バリデーションデータに対する損失関数値のリスト、
            評価指標のリスト
        """
        # ラン名
        validation = i_fold != "all"

        for i, model in enumerate(self.models):
            # 学習データに対するカスタム損失関数とカスタム評価関数をセット
            if model.make_loss_func is not None:
                model.loss_func_tr = model.make_loss_func(tr_z)

            if validation:
                if model.make_loss_func is not None:
                    # バリデーションデータに対するカスタム損失関数をセット
                    model.loss_func_va = model.make_loss_func(va_z)

                if model.make_eval_func is not None:
                    # バリデーションデータに対するカスタム評価関数をセット
                    model.eval_func_va = model.make_eval_func(va_z)

                # Optuna で最適なハイパーパラメータ探索
                sampler = optuna.samplers.TPESampler(seed=random_seed)
                study = optuna.create_study(sampler=sampler, direction="minimize")

                try:
                    study.optimize(
                        lambda trial: self._objective(
                            trial,
                            model,
                            tr_x,
                            tr_y,
                            va_x,
                            va_y,
                        ),
                        n_trials=n_trials,
                    )

                except Exception as e:
                    print(f"Optimization for model {i} failed with exception:\n{e}")
                    continue

                # 最適なハイパーパラメータで学習
                try:
                    model.train(tr_x, tr_y, va_x, va_y)
                except Exception as e:
                    print(
                        f"Training for model {i} with validation failed with exception:\n{e}"
                    )
                    continue

            else:
                try:
                    # 最適なハイパーパラメータで学習
                    model.train(tr_x, tr_y, None, None, None)
                except Exception as e:
                    print(f"Training for model {i} failed with exception:\n{e}")
                    continue

            # モデルを保存
            model.model_name += f"_{i_fold}"
            model.save_model()

    def _objective(
        self,
        trial: optuna.trial.Trial,
        model: _Model,
        tr_x: pd.DataFrame,
        tr_y: pd.DataFrame,
        va_x: pd.DataFrame,
        va_y: pd.DataFrame,
    ):
        """
        Optuna を用いたハイパーパラメータ探索の目的関数。
        訓練後に、バリデーションデータに対する損失関数値を返す。

        Parameters
        ----------
        trial : optuna.trial.Trial
            Optuna が最適化するハイパーパラメータ

        model : _Model
            モデルのクラス

        tr_x : pd.DataFrame
            訓練データの特徴量

        tr_y : pd.DataFrame
            訓練データの目的変数

        va_x : pd.DataFrame
            バリデーションデータの特徴量

        va_y : pd.DataFrame
            バリデーションデータの目的変数

        Returns
        -------
        float
            バリデーションデータに対する損失関数値
        """
        model.set_params(trial)
        model.train(tr_x, tr_y, va_x, va_y)
        va_pred = model.predict(va_x)
        return model.loss(va_y, va_pred)

    def run_train_all(self) -> pd.DataFrame:
        """
        学習データすべてで学習し、そのモデルを保存。
        """
        logger.info(f"{self.run_name} - start training all")
        df = self._train_fold("all")
        logger.info(f"{self.run_name} - end training all")

        # 評価結果を保存
        logger.result_scores(self.run_name, df)

        return df

    def run_predict_cv(self) -> None:
        """
        クロスバリデーションで学習した各フォールドのモデルの平均により、テストデータを予測する

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
        DumperLoader.dump(pred_avg, f"../model/pred/{self.run_name}-test.pkl")

        logger.info(f"{self.run_name} - end prediction cv")

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
        DumperLoader.dump(pred, f"../model/pred/{self.run_name}-test.pkl")

        logger.info(f"{self.run_name} - end prediction all")

    def load_index_fold(
        self,
        i_fold: int,
    ) -> np.array:
        """
        クロスバリデーションでの fold 番号を指定して、対応するレコードのインデックスを返す

        Parameters
        ----------
        i_fold : int
            fold 番号

        Returns
        -------
        np.array
            fold に対応するレコードのインデックス
        """
        # インデックス保存用のファイルパスを指定
        index_file_path = f"../model/{self.run_name}/index_fold.pkl"

        # すでにインデックスファイルが存在する場合、それをロード
        if os.path.exists(index_file_path):
            with open(index_file_path, "rb") as f:
                fold_indices = pickle.load(f)

        else:
            # 学習データ・バリデーションデータを分けるインデックスを返す
            train = DumperLoader.load_train(self.run_name)
            train_y = train[self.target]
            dummy_x = np.zeros(len(train_y))
            cv = Splitter.k_fold(self.task, self.n_fold)
            fold_indices = list(cv.split(dummy_x, train_y))

            # インデックスをファイルとして保存
            with open(index_file_path, "wb") as f:
                pickle.dump(fold_indices, f)

        return fold_indices[i_fold]
