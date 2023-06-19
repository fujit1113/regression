# 標準ライブラリ
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

# サードパーティのライブラリ
import numpy as np
import pandas as pd
import optuna
from tqdm import tqdm

# ローカルのライブラリ / アプリケーション
from CPoC.config.config import config
from CPoC.models.model import _Model
from CPoC.utils.dumper_loader import DumperLoader
from CPoC.utils.logger import Logger
from .splitter import Splitter
from .validator import Validator


logger = Logger()


class Trainer:
    def __init__(self, validator: Validator):
        """
        コンストラクタ

        Parameters
        ----------
        validator : Validator
            条件: アトリビュート is_available が True
        """
        # 確認: validator で入力確認済み
        if validator._is_available:
            self.bunch_name = validator.bunch_name
            self.task = validator.task
            self.target = validator.target
            self.additions = validator.additions
            self.features = validator.features
            self.n_fold = validator.n_fold
            self.n_trials = validator.n_trials
            self.random_seed = validator.random_seed
            self.description = validator.description

            self.models = validator.models

            path_model = Path(config["PATH_MODEL"]) / self.bunch_name
            path_model.mkdir(exist_ok=True)

        else:
            raise ValueError(
                "The provided Validator instance is not available. Please make sure the 'is_available' attribute of your Validator instance is set to True."
            )

    def train_cv(self) -> pd.DataFrame:
        """
        処理: 交差検証を実行\n
        仕様1: 損失関数値と評価関数値を出力\n
        仕様2: 各モデルのハイパーパラメータを保存\n
        """
        # 仕様: 交差検証の開始をログに残す
        logger.info(f"{self.bunch_name} - start training cv")

        # 処理: 各フォールドで学習
        for i_fold in range(self.n_fold):
            logger.info(f"{self.bunch_name} fold {i_fold} - start training")
            self._train_fold(i_fold)
            logger.info(f"{self.bunch_name} fold {i_fold} - end training")

        # 仕様: 交差検証の終了をログに残す
        logger.info(f"{self.bunch_name} - end training cv")

    def _train_fold(
        self,
        i_fold: Union[int, str],
    ) -> None:
        """
        処理: フォールドを指定して該当するインデックスを取得

        Parameters
        ----------
        i_fold : Union[int, str]
            フォールドの番号（すべてのときには'all'とする)

        Returns
        -------
        pd.DataFrame
            モデル名、損失関数値、評価指標をカラムにもつデータフレーム
        """
        # 処理: 学習データをロード
        is_validation = i_fold != "all"
        learn = DumperLoader.load_learn(self.bunch_name)
        learn_x = learn[self.features]
        learn_y = learn[self.target]
        learn_z = learn[self.additions] if self.additions else None

        if is_validation:
            # 処理: 訓練データと検証データを取得
            train_idx, valid_idx = self.load_index_fold(i_fold)
            train_x, train_y = (learn_x.iloc[train_idx], learn_y.iloc[train_idx])
            valid_x, valid_y = (learn_x.iloc[valid_idx], learn_y.iloc[valid_idx])

            if learn_z:
                train_z, valid_z = (learn_z.iloc[train_idx], learn_z.iloc[valid_idx])
            else:
                train_z, valid_z = (None, None)

            self._train_models(
                i_fold, train_x, train_y, train_z, valid_x, valid_y, valid_z
            )

        else:
            # 仕様: 全学習データで学習
            self._train_models(i_fold, learn_x, learn_y, learn_z)

    def _train_models(
        self,
        i_fold: Union[int, str],
        train_x: pd.DataFrame,
        train_y: pd.DataFrame,
        train_z: Optional[pd.DataFrame] = None,
        valid_x: Optional[pd.DataFrame] = None,
        valid_y: Optional[pd.DataFrame] = None,
        valid_z: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        処理: 最適なハイパーパラメータを探索し、モデルをつくる\n
        仕様: ハイパーパラメータを出力する

        Parameters
        ----------
        i_fold : Union[int, str]
            交差検証のフォールド番号 (全データを使う場合は 'all')

        train_x : pd.DataFrame
            訓練データの特徴量

        train_y : pd.DataFrame
            訓練データの目的変数

        train_z : Optional[pd.DataFrame]
            訓練データのカスタム損失関数やカスタム評価関数に必要な変数

        valid_x : Optional[pd.DataFrame]
            検証データの特徴量

        valid_y : Optional[pd.DataFrame]
            検証データの目的変数

        valid_z : Optional[pd.DataFrame]
            検証データのカスタム損失関数やカスタム評価関数に必要な変数
        """
        # バンチ名
        is_validation = i_fold != "all"

        for model in self.models:
            # 処理1: 損失関数と評価関数を設定
            Trainer._setup_loss_and_eval(model, is_validation, train_z, valid_z)

            # 処理2: Optuna で最適なハイパーパラメータを探索
            if is_validation:
                tuned_params, loss_score_valid = Trainer._tuning_with_optuna(
                    model,
                    train_x,
                    train_y,
                    valid_x,
                    valid_y,
                    self.random_seed,
                    self.n_trials,
                )
                model.set_params(tuned_params)

                # 処理3: モデルを学習
                model.train(train_x, train_y, valid_x, valid_y)

                # 仕様1: 訓練データに対する損失関数値を取得
                preds = model.predict(train_x)
                loss_score_train = model.loss_func_optuna_train(preds, train_y)

            else:
                model.train(train_x, train_y, None, None)
                loss_score_valid = None
                loss_score_train = None

            # 処理4: カスタム損失関数とカスタム評価関数を初期化
            model.initialize_funcs()

            # 仕様2: モデルを保存
            model_name = model.model_name + f"_{i_fold}"
            model.save_model(model_name)

            # 仕様3: result.log に結果を出力
            logger.result(f"{model_name}\t{loss_score_train}\t{loss_score_valid}")

    @staticmethod
    def _setup_loss_and_eval(
        model: _Model,
        is_validation: bool,
        train_z: Optional[pd.DataFrame],
        valid_z: Optional[pd.DataFrame],
    ) -> None:
        """
        処理: 損失関数と損失関数を設定

        Parameters
        ----------
        model : _Model
            モデル

        is_validation : bool
            検証か否か

        train_z : Optional[pd.DataFrame]
            訓練データのカスタム損失関数やカスタム評価関数に必要な変数

        va_z : Optional[pd.DataFrame]
            検証データのカスタム損失関数やカスタム評価関数に必要な変数
        """
        # 確認: カスタム損失関数を宣言しているか否か
        if model.make_loss_func:
            # 処理: 訓練用のカスタム損失関数をつくる
            model.loss_func = (
                model.make_loss_func(False, train_z)
                if train_z
                else model.make_loss_func(False)
            )

            if is_validation:
                # 処理: optuna 用のカスタム損失関数をつくる
                model.loss_func_optuna_valid = (
                    model.make_loss_func(True, valid_z)
                    if valid_z
                    else model.make_loss_func(True)
                )

                # 仕様: 訓練データに対する optuna 用のカスタム損失関数もつくる
                # 理由: 過学習していないか確認するため
                model.loss_func_optuna_tr = (
                    model.make_loss_func(True, train_z)
                    if train_z
                    else model.make_loss_func(True)
                )
        else:
            # 処理: デフォルト損失関数に応じた optuna 用の損失関数をつくる
            model.loss_func_optuna_valid = model.default_loss_func_optuna()
            model.loss_func_optuna_train = model.default_loss_func_optuna()

        # 処理: カスタム評価関数を宣言しているとき
        if model.make_eval_func:
            # 処理: 検証データに対するカスタム評価関数をセット
            model.eval_func = (
                model.make_eval_func(valid_z) if valid_z else model.make_eval_func()
            )

    @staticmethod
    def _tuning_with_optuna(
        model: _Model,
        train_x: pd.DataFrame,
        train_y: pd.DataFrame,
        valid_x: pd.DataFrame,
        valid_y: pd.DataFrame,
        random_seed: int = config["RANDOM_SEED"],
        n_trials: int = config["N_TRIALS"],
    ) -> Dict[str, Any]:
        """
        処理: optuna で最適なハイパーパラメータを探索

        Parameters
        ----------
        model : _Model
            モデル

        train_x : pd.DataFrame
            訓練データの特徴量

        train_y : pd.DataFrame
            訓練データの目的変数

        valid_x : pd.DataFrame
            検証データの特徴量

        valid_y : pd.DataFrame
            検証データの目的変数

        Returns
        -------
        Tuple[Dict[str, Any], float]
            最適なハイパーパラメータの辞書とそのときの損失関数値
        """
        # 仕様: Tree-structured Parzen Estimator（TPE）
        sampler = optuna.samplers.TPESampler(seed=random_seed)
        study = optuna.create_study(sampler=sampler, direction="minimize")

        study.optimize(
            lambda trial: Trainer._objective(
                trial,
                model,
                train_x,
                train_y,
                valid_x,
                valid_y,
            ),
            n_trials=n_trials,
        )

        return study.best_params, study.best_value

    @staticmethod
    def _objective(
        trial: optuna.trial.Trial,
        model: _Model,
        train_x: pd.DataFrame,
        train_y: pd.DataFrame,
        valid_x: pd.DataFrame,
        valid_y: pd.DataFrame,
    ):
        """
        Optuna を用いたハイパーパラメータ探索の目的関数。\n
        学習後に、検証データに対する損失関数値を返す。

        Parameters
        ----------
        trial : optuna.trial.Trial
            Optuna が最適化するハイパーパラメータ

        model : _Model
            モデルのクラス

        train_x : pd.DataFrame
            訓練データの特徴量

        train_y : pd.DataFrame
            訓練データの目的変数

        valid_x : pd.DataFrame
            検証データの特徴量

        valid_y : pd.DataFrame
            検証データの目的変数

        Returns
        -------
        float
            検証データに対する損失関数値
        """
        model.suggest_params(trial)
        model.train(train_x, train_y, valid_x, valid_y)
        preds = model.predict(valid_x)
        return model.loss_func_optuna_valid(preds, valid_y)

    def run_train_all(self) -> pd.DataFrame:
        """
        学習データすべてで学習し、そのモデルを保存。
        """
        logger.info(f"{self.bunch_name} - start training all")
        df = self._train_fold("all")
        logger.info(f"{self.bunch_name} - end training all")

        # 評価結果を保存
        logger.result_scores(self.bunch_name, df)

        return df

    def run_predict_cv(self) -> None:
        """
        交差検証で学習した各フォールドのモデルの平均により、テストデータを予測する

        あらかじめrun_train_cvを実行しておく必要がある
        """
        logger.info(f"{self.bunch_name} - start prediction cv")

        test_x = self.load_x_test()

        preds = []

        # 各foldのモデルで予測
        for i_fold in tqdm(range(self.n_fold), desc="Processing folds"):
            logger.info(f"{self.bunch_name} - start prediction fold:{i_fold}")
            model = self.build_model(i_fold)
            model.load_model()
            pred = model.predict(test_x)
            preds.append(pred)
            logger.info(f"{self.bunch_name} - end prediction fold:{i_fold}")

        # 予測の平均値を出力
        pred_avg = np.mean(preds, axis=0)

        # 予測結果の保存
        DumperLoader.dump(pred_avg, f"../model/pred/{self.bunch_name}-test.pkl")

        logger.info(f"{self.bunch_name} - end prediction cv")

    def run_predict_all(self) -> None:
        """学習データすべてで学習したモデルでテストデータを予測

        あらかじめrun_train_allを実行しておくこと
        """
        logger.info(f"{self.bunch_name} - start prediction all")

        test_x = self.load_x_test()

        # 学習データすべてで学習したモデルで予測
        i_fold = "all"
        model = self.build_model(i_fold)
        model.load_model()
        pred = model.predict(test_x)

        # 予測結果の保存
        DumperLoader.dump(pred, f"../model/pred/{self.bunch_name}-test.pkl")

        logger.info(f"{self.bunch_name} - end prediction all")

    def load_index_fold(
        self,
        i_fold: int,
    ) -> np.array:
        """
        交差検証での fold 番号を指定して、対応するレコードのインデックスを返す

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
        index_file_path = (
            Path(config["PATH_MODEL"]) / self.bunch_name / "index_fold.pkl"
        )

        # すでにインデックスファイルが存在する場合、それをロード
        if index_file_path.exists():
            with open(index_file_path, "rb") as f:
                fold_indices = pickle.load(f)

        else:
            # 学習データと検証データに分けるインデックスを返す
            train = DumperLoader.load_train(self.bunch_name)
            train_y = train[self.target]
            dummy_x = np.zeros(len(train_y))
            cv = Splitter.k_fold(self.task, self.n_fold)
            fold_indices = list(cv.split(dummy_x, train_y))

            # インデックスをファイルとして保存
            with open(index_file_path, "wb") as f:
                pickle.dump(fold_indices, f)

        return fold_indices[i_fold]
