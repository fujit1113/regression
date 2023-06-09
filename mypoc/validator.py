# 標準ライブラリ
import json
import os
from inspect import Parameter, signature
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import warnings
from pathlib import Path

# サードパーティのライブラリ
import numpy as np
import pandas as pd

# ローカル（自身のプロジェクト）のライブラリ
from .model import _Model
from mypoc.config.config import config
from mypoc.utils.checker import Checker
from mypoc.utils.dumper_loader import DumperLoader


class Validator:
    """
    Runner 実行前に、各パラメータの妥当性を確認する

    Attributes
    ----------
    run_name : str
        ランの名前

    task : str
        タスクの種類 ('regression' か 'binary' か 'multiclass')

    target : str
        目的変数

    features : List[str]
        特徴量

    recipes : List[Dict]
        モデルのレシピ

    additions : Optional[List[str]]
        カスタム損失関数やカスタム評価関数に必要なカラム

    n_fold : int
        交差検証のフォールド数

    description : str
        ランの説明

    Methods
    -------
    validate_attributes():
        ※ コンストラクタ作成後に実行してください
    """

    def __init__(
        self,
        run_name: str,
        task: str,
        target: str,
        features: List[str],
        recipes: List[Dict[str, Any]],
        additions: Optional[List[str]] = None,
        n_fold: int = config["N_FOLD"],
        description: str = "",
    ):
        """
        コンストラクタ

        Parameters
        ----------
        run_name : str
            ランの名前

        task : str
            タスクの種類 ('regression' か 'binary' か 'multiclass')

        target : str
            目的変数

        features : List[str]
            特徴量

        recipes: List[Dict]
            モデルのレシピ

        additions : Optional[List[str]]
            損失関数や評価関数に必要なカラム

        n_fold: int
            交差検証のフォールド数

        description : str
            ランの説明
        """
        # 仕様: 初期化
        self.run_name = run_name
        self.task = task
        self.target = target
        self.features = features
        self.recipes = recipes
        self.additions = additions
        self.n_fold = n_fold
        self.description = description

        # 仕様: are_attrs_avaible() 実行後に True
        self._is_available = False

    def __repr__(self):
        """
        仕様: インスタンス名だけ呼ばれたとき、アトリビュートの値を表示
        """
        original_repr = super().__repr__()
        attrs = "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())

        return f"{original_repr}\n{attrs}"

    def validate_attributes(self) -> None:
        """
        確認: 各アトリビュートは妥当
        """
        # 確認1: 各アトリビュートは妥当
        self.check_run_files_exit(self.run_name)
        self.check_task(self.task)
        self.check_target_features_additions(
            self.target, self.features, self.additions, self.run_name
        )
        self.check_n_fold(self.n_fold)
        self.check_description(self.description)

        # 確認2: recipes は妥当 (+ 欠損しているキーと値を補う)
        self.recipes = self.check_recipes(self.recipes, self.run_name, self.additions)

        # 仕様1: 未学習モデルを作成
        self.models = self.build_models(self.task, self.recipes)

        # 仕様2: 妥当性確認を完了
        self._is_available = True

    def save_to_JSON(self, filename: str):
        """
        仕様: アトリビュートとその値を JSON 形式で保存

        Parameters
        ----------
        filename : str
            ファイル名
        """
        # 確認: 妥当性確認ずみ
        if self._is_available:
            with open(filename, "w") as f:
                f.write(self._save_to_JSON())
        else:
            raise ValueError(
                "All attributes are not available. The object was not saved."
            )

    def _save_to_JSON(self):
        """
        仕様: アトリビュートとその値を JSON 形式に変換する
        """
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    @staticmethod
    def check_run_files_exit(run_name: str) -> None:
        """
        確認: 指定のラン関係のフォルダに必要なファイルがある

        Parameters
        ----------
        run_name : str
            ランの名前
        """
        # 確認1: run_name は空文字でない
        if not run_name:
            raise ValueError("The run name should not be an empty string.")

        path_input = Path(config["PATH_INPUT"]) / run_name
        path_train = path_input / "train.pkl"
        path_test = path_input / "test.pkl"
        path_model = Path(config["PATH_MODEL"]) / run_name

        # 確認2: ランのインプットフォルダが存在
        Checker.check_path_exists(path_input)

        # 確認3: ランのインプットフォルダに学習データが存在
        Checker.check_path_exists(path_train)

        # 確認4: ランのインプットフォルダにテストデータが存在
        Checker.check_path_exists(path_test)

        # 確認5: ランのモデルフォルダが存在するとき、モデルを追加
        if os.path.exists(path_model):
            warnings.warn(
                f"The model folder for the run '{run_name}' already exists. New models will be added to this folder."
            )

    @staticmethod
    def check_task(task: str) -> None:
        """
        確認: task は妥当

        Parameters
        ----------
        task : str
            タスクの種類 ('regression' か 'binary' か 'multiclass')
        """
        # 確認1: config["TASKS"] は task を含む
        attrs = ", ".join(val for val in config["TASKS"])
        Checker.check_list_inclusion([task], "task", config["TASKS"], f"[{attrs}]")

    @staticmethod
    def check_target_features_additions(
        target: str,
        features: List[str],
        additions: Optional[List[str]],
        run_name: str,
    ) -> None:
        """
        確認: target と features と additions は妥当

        Parameters
        ----------
        target : str
            目的変数

        features : List[str]
            特徴量

        additions : Optional[List[str]]
            カスタム損失関数やカスタム評価関数に必要なカラム

        run_name : str
            ランの名前
        """
        # 準備: 学習データとテストデータのカラム名のリスト
        train_col = list(DumperLoader.load_train(run_name).columns)
        test_col = list(DumperLoader.load_test(run_name).columns)

        # 確認1: target は str
        Checker.check_param_type(target, "target", str)

        # 確認2: 学習データとテストデータのカラムは target を含む
        Checker.check_list_inclusion([target], "target", train_col, "train")
        Checker.check_list_inclusion([target], "target", test_col, "test")

        # 確認3: features は List、その各要素は str
        Checker.check_list_elements_type(features, "features", str)

        # 確認4: features と target は重複要素なし
        Checker.check_no_common_elements(target, "target", features, "features")

        # 確認5: 学習データとテストデータのカラムは features を含む
        Checker.check_list_inclusion(features, "features", train_col, "train")
        Checker.check_list_inclusion(features, "features", test_col, "test")

        if additions:
            # 確認6: additions の各要素は str
            Checker.check_list_elements_type(additions, "additions", str)

            # 確認7: additions は target は重複要素なし
            Checker.check_no_common_elements(target, "target", additions, "additions")

            # 確認8: 学習データとテストデータは additions を含む
            Checker.check_list_inclusion(additions, "additions", train_col, "train")
            Checker.check_list_inclusion(additions, "additions", test_col, "test")

    @staticmethod
    def check_recipes(
        recipes: List[Dict[str, Any]],
        run_name: str,
        additions: Optional[List[str]],
    ) -> List[Dict]:
        """
        確認: recipes は妥当

        Parameters
        ----------
        recipes : List[Dict[str, Any]]
            モデルのレシピ

        Returns
        -------
        List[Dict[str, Any]]
            欠損情報を補ったモデルのレシピ
        """
        # 確認1: recipes は List、その要素は Dict
        Checker.check_list_elements_type(recipes, "recipes", dict)

        # 仕様1: 欠損情報を補ったモデルのレシピを作成
        recipes_full = []
        for i, recipe in enumerate(recipes):
            recipe_full = {}

            # 確認2: 各キーに対する値は妥当
            recipe_full["model_class"] = Validator.check_model_class(recipe)
            recipe_full["model_name"] = Validator._check_model_name(recipe, i)
            recipe_full["fixed_params"] = Validator.check_fixed_params(recipe)
            recipe_full["make_loss_func"] = Validator.ensure_valid_make_func(
                recipe, run_name, additions, "make_loss_func"
            )
            recipe_full["make_eval_func"] = Validator.ensure_valid_make_func(
                recipe, run_name, additions, "make_eval_func"
            )

            # 仕様2: リストに追加
            recipes_full.append(recipe_full)

        # 確認2: recipes_full は同じモデル名を含まない
        Checker.check_duplicate_keys(recipes_full, "model_name")

        return recipes_full

    @staticmethod
    def check_model_class(recipe: Dict[str, Any]) -> Callable[[], _Model]:
        """
        確認: model_class は妥当

        Parameters
        ----------
        recipe : Dict
            モデルのレシピ

        Returns
        -------
        Callable[[], _Model]
        """
        # 確認1: recipes は 'model_class' を含む
        Checker.check_key_in_dict("model_class", recipe)
        model_class = recipe["model_class"]

        # 確認2: model_class は '_Model' の派生クラス
        Checker.check_inheritance(model_class, _Model)

        # 確認3: model_class はアトリビュート _param_space を含む
        Checker.check_has_attribute(model_class, "_param_space")

        return model_class

    @staticmethod
    def _check_model_name(
        recipe: Dict[str, Any],
        i: int,
    ) -> str:
        """
        確認: model_name は妥当

        Parameters
        ----------
        recipe : Dict[str, Any]
            モデルのレシピ

        i : int
            要素番号 (無名のモデル名のサフィックスに使う)

        Returns
        -------
        str
        """
        if "model_name" in recipe:
            # 確認: model_name は str
            Checker.check_param_type(recipe["model_name"], "model_name", str)
            return recipe["model_name"]
        else:
            return f"noname_{i}"

    @staticmethod
    def check_fixed_params(recipe: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        確認: fixed_params は妥当

        Parameters
        ----------
        recipe : Dict[str, Any]
            モデルのレシピ

        Returns
        -------
        Optional[Dict[str, Any]]
        """
        if "fixed_params" in recipe:
            fixed_params = recipe["fixed_params"]
            model_class = recipe["model_class"]

            # 確認1: fixed_params は Dict
            Checker.check_param_type(fixed_params, "fixed_params", Dict)

            # 確認2: model_class の _param_space のキーは、 fixed_params の各キーを含む
            Checker.check_list_inclusion(
                list(fixed_params.keys()), list(model_class._param_space.keys())
            )

            return fixed_params

        else:
            return None

    @staticmethod
    def ensure_valid_make_func(
        recipe: Dict[str, Any],
        run_name: str,
        additions: Optional[List[str]],
        make_func_key: str,
    ) -> Optional[Callable]:
        """
        確認: 指定の関数 ('make_loss_func' か 'make_eval_func') は妥当

        Parameters
        ----------
        recipe : Dict[str, Any]
            モデルのレシピ

        run_name : str
            ランの名前

        additions : Optional[List[str]]
            損失関数や評価関数に必要なカラム

        make_func_key : str
            検証する関数のキー名 ('make_loss_func' か 'make_eval_func')

        Returns
        -------
        Optional[Callable]
        """
        if make_func_key in recipe:
            model_class = recipe["model_class"]
            make_func = recipe[make_func_key]

            # 確認1: make_func は Callable
            Checker.check_callable(make_func, f"{make_func_key}")

            # 処理: make_func の引数と戻り値情報を取得
            sig = signature(make_func)
            params = sig.parameters

            # 確認2: エンクロージャの引数は1つ、もしくは、持たない
            if len(params) > 1:
                raise TypeError(
                    f"Function {make_func.__name__} should have one or no argument."
                )

            # 確認3: エンクロージャの引数は pd.DataFrame
            if len(params) == 1:
                for name, param in params.items():
                    if (
                        param.annotation != Parameter.empty
                        and param.annotation is not pd.DataFrame
                    ):
                        raise TypeError(
                            f"Argument '{name}' of function {make_func.__name__} should be of type pd.DataFrame."
                        )

                # 処理: クロージャを生成
                df_trial = DumperLoader.load_train(run_name)[additions].head(5)
                func = make_func(df_trial)
            else:
                func = make_func()

            # 確認4: クロージャの型は妥当
            sig = signature(func)
            if make_func_key == "make_loss_func":
                model_class._check_loss_func(sig)
            elif make_func_key == "make_eval_func":
                model_class._check_eval_func(sig)

            return make_func

        else:
            return None

    @staticmethod
    def build_models(
        task: str,
        recipes: List[Dict[str, Any]],
    ) -> List[_Model]:
        """
        仕様: recipes にもとづいて、モデルリストを作る

        Parameters
        ----------
        task : str
            タスクの種類 ('regression' か 'binary' か 'multiclass')

        recipes : List[Dict]
            モデルレシピ

        Returns
        -------
        List[_Model]
        """
        models = []
        for recipe in recipes:
            # 仕様: モデルのインスタンスをリストに追加
            models.append(
                recipe["model_class"](
                    task,
                    recipe["model_name"],
                    recipe["fixed_params"],
                    recipe["make_loss_func"],
                    recipe["make_eval_func"],
                )
            )

        return models

    @staticmethod
    def check_description(description: str) -> None:
        """
        確認: description は妥当

        Parameters
        ----------
        description: str
            ランの説明
        """
        # 確認1: description は str
        Checker.check_param_type(description, "description", str)

    @staticmethod
    def check_n_fold(n_fold: int):
        """
        確認: n_fold は妥当

        Parameters
        ----------
        n_fold : int
            交差検証のフォールド数
        """
        # 確認: n_fold は自然数
        Checker.check_natural_number(n_fold, "n_fold")

    @property
    def is_available(self):
        return self._is_available
