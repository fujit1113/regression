# 標準ライブラリ
import json
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path

# サードパーティのライブラリ
import numpy as np
import pandas as pd

# ローカルのライブラリ
from CPoC.config.config import config
from CPoC.models.model import _Model
from CPoC.utils.checker import Checker
from CPoC.utils.dumper_loader import DumperLoader


class Validator:
    """
    Trainer 実行前に、各パラメータの妥当性を確認する

    Attributes
    ----------
    bunch_name : str
        バンチの名前

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

    n_trials : int
        交差検証のフォールド数

    random_seed : int
        乱数生成のシード

    description : str
        バンチの説明

    Methods
    -------
    validate_attributes():
        ※ コンストラクタ作成後に実行してください
    """

    def __init__(
        self,
        bunch_name: str,
        task: str,
        target: str,
        features: List[str],
        recipes: List[Dict[str, Any]],
        additions: List[str] = [],
        n_fold: int = config["N_FOLD"],
        n_trials: int = config["N_TRIALS"],
        random_seed: int = config["RANDOM_SEED"],
        description: str = "",
    ):
        """
        コンストラクタ

        Parameters
        ----------
        bunch_name : str
            バンチの名前

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

        n_trials : int
            Optuna のハイパーパラメータの試行回数

        random_seed : int
            乱数生成のシード

        description : str
            バンチの説明
        """
        # 仕様: 初期化
        self.bunch_name = bunch_name
        self.task = task
        self.target = target
        self.features = features
        self.recipes = recipes
        self.additions = additions
        self.n_fold = n_fold
        self.n_trials = n_trials
        self.random_seed = random_seed
        self.description = description

        # 仕様: validate_attributes() 実行後に True
        self._is_available = False

    def __repr__(self):
        """
        仕様: インスタンス名だけ呼ばれたとき、アトリビュートの値を表示
        """
        original_repr = super().__repr__()

        attr_lines = []
        for k, v in self.__dict__.items():
            if isinstance(v, list) and all(isinstance(i, dict) for i in v):
                for i, dict_item in enumerate(v):
                    for dict_k, dict_v in dict_item.items():
                        attr_lines.append(f"{k}[{i}][{dict_k}]: {dict_v}")
            else:
                attr_lines.append(f"{k}: {v}")

        attrs = "\n".join(attr_lines)

        return f"{original_repr}\n{attrs}"

    def check_attributes(self) -> None:
        """
        確認: 各アトリビュートは妥当
        """
        # 確認1: bunch_name は妥当
        self.check_bunch_name(self.bunch_name)

        # 確認2: task は妥当
        self.check_task(self.task)

        # 確認3: target は妥当
        self.check_target(self.target, self.bunch_name)

        # 確認4: features は妥当
        self.check_features(self.features, self.target, self.bunch_name)

        # 確認5: additions が None でないとき、additions は妥当
        if self.additions:
            self.check_additions(self.additions, self.target, self.bunch_name)

        # 確認6: n_fold は妥当
        self.check_n_fold(self.n_fold)

        # 確認7: n_trials は妥当
        self.check_n_trials(self.n_trials)

        # 確認8: random_seed は妥当
        self.check_random_seed(self.random_seed)

        # 確認9: description は妥当
        self.check_description(self.description)

        # 確認10: recipes は妥当 + 処理1: 欠損しているキーと値を補う
        self.recipes = self.check_recipes(self.recipes, self.bunch_name, self.additions)

        # 仕様2: recipes にもとづいて、未学習モデルをつくる
        self.models = self.build_models(self.bunch_name, self.task, self.recipes)

        # 仕様3: 妥当性確認を完了
        self._is_available = True
        print(
            "Instance validated successfully. It is safe to pass this instance to the Trainer's constructor."
        )

    @staticmethod
    def check_bunch_name(bunch_name: str) -> None:
        """
        確認: bunch_name は妥当

        Parameters
        ----------
        bunch_name : str
            バンチの名前
        """
        # 確認1: bunch_name は str
        Checker.check_param_type(bunch_name, "bunch_name", str)

        # 確認2: bunch_name は空文字ではない
        Checker.check_existence(bunch_name, "bunch_name")

        # 処理: パスをつくる
        path_input = Path(config["PATH_INPUT"]) / bunch_name
        path_learn = path_input / "learn.pkl"
        path_test = path_input / "test.pkl"

        # 確認3: バンチのインプットフォルダがある
        Checker.check_path_existence(path_input)

        # 確認4: バンチのインプットフォルダに学習データがある
        Checker.check_path_existence(path_learn)

        # 確認5: バンチのインプットフォルダにテストデータがある
        Checker.check_path_existence(path_test)

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
        Checker.check_element_inclusion(
            task, config["TASKS"], "task of your input", "tasks to be inputted"
        )

    @staticmethod
    def check_target(
        target: str,
        bunch_name: str,
    ) -> None:
        """
        確認: target は妥当

        Parameters
        ----------
        target : str
            目的変数

        bunch_name : str
            バンチの名前
        """
        # 準備: 学習データとテストデータのカラム名のリスト
        learn_col = DumperLoader.load_learn(bunch_name).columns
        test_col = DumperLoader.load_test(bunch_name).columns

        # 確認1: target は str
        Checker.check_param_type(target, "target", str)

        # 確認2: 学習データのカラムは target を含む
        Checker.check_element_inclusion(
            target, learn_col, "target of your input", "the learn-DataFrame columns"
        )

        # 確認3: テストデータのカラムは target を含む
        Checker.check_element_inclusion(
            target, test_col, "target of your input", "the test-DataFrame columns"
        )

    @staticmethod
    def check_features(
        features: List[str],
        target: str,
        bunch_name: str,
    ) -> None:
        """
        確認: features は妥当

        Parameters
        ----------
        features : List[str]
            特徴量

        target : str
            目的変数

        bunch_name : str
            バンチの名前
        """
        Validator._check_columns(features, "features of your input", target, bunch_name)

    @staticmethod
    def check_additions(
        additions: List[str],
        target: str,
        bunch_name: str,
    ) -> None:
        """
        確認: additions は妥当

        Parameters
        ----------
        additions : List[str]
            カスタム損失関数やカスタム評価関数に必要なカラム

        target : str
            目的変数

        bunch_name : str
            バンチの名前
        """
        Validator._check_columns(
            additions, "additions of your input", target, bunch_name
        )

    @staticmethod
    def _check_columns(
        columns: List[str],
        columns_name: str,
        target: str,
        bunch_name: str,
    ) -> None:
        """
        確認: 指定のカラムが妥当

        Parameters
        ----------
        columns : List[str]
            チェックするカラム\n
            特徴量のカラムかカスタム損失関数、カスタム評価関数に必要なカラム

        columns_name : str
            チェックするカラムの名前

        target : str
            目的変数

        bunch_name : str
            バンチの名前
        """
        # 準備: 学習データとテストデータのカラム名のリスト
        learn_col = list(DumperLoader.load_learn(bunch_name).columns)
        test_col = list(DumperLoader.load_test(bunch_name).columns)

        # 確認1: columns は List、その各要素は str
        Checker.check_list_elements_type(columns, columns_name, str)

        # 確認2: columns は空リストではない
        Checker.check_existence(columns)

        # 確認3: columns に重複する要素はない
        Checker.check_no_duplication(columns, columns_name)

        # 確認4: columns は target を含まない
        Checker.check_element_exclusion(target, columns, "target", columns_name)

        # 確認5: 学習データのカラムは columns を含む
        Checker.check_list_subset(
            columns, learn_col, columns_name, "the learn-DataFrame columns"
        )

        # 確認6: テストデータのカラムは columns を含む
        Checker.check_list_subset(
            columns, test_col, columns_name, "the test-DataFrame columns"
        )

    @staticmethod
    def check_recipes(
        recipes: List[Dict[str, Any]],
        bunch_name: str,
        additions: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        確認: recipes は妥当

        Parameters
        ----------
        recipes : List[Dict[str, Any]]
            モデルのレシピ

        bunch_name : str
            バンチの名前

        additions : List[str]
            カスタム損失関数やカスタム評価関数に必要なカラム

        Returns
        -------
        List[Dict[str, Any]]
            欠損情報を補ったモデルのレシピ
        """
        # 確認1: recipes は List、その要素は Dict
        Checker.check_list_elements_type(recipes, "recipes", dict)

        # 仕様1: 欠損情報を補ったモデルのレシピを作成
        recipes_full = []
        for recipe in recipes:
            recipe_full = {}

            # 確認2: 各キーに対する値は妥当
            recipe_full["model_class"] = Validator.check_model_class(recipe)
            recipe_full["model_name"] = Validator.check_model_name(recipe)
            recipe_full["fixed_params"] = Validator.check_fixed_params(recipe)
            recipe_full["make_loss_func"] = Validator.check_make_loss_func(
                recipe, bunch_name, additions
            )
            recipe_full["make_eval_func"] = Validator.check_make_eval_func(
                recipe, bunch_name, additions
            )

            # 仕様2: リストに追加
            recipes_full.append(recipe_full)

        # 確認2: recipes_full は同じモデル名を含まない
        Checker.check_value_no_duplication(
            recipes_full, "recipe of your input", "model_name"
        )

        return recipes_full

    @staticmethod
    def check_model_class(recipe: Dict[str, Any]) -> Callable[..., _Model]:
        """
        確認: model_class は妥当

        Parameters
        ----------
        recipe : Dict
            モデルのレシピ

        Returns
        -------
        Callable[..., _Model]
        """
        # 確認1: recipes は model_class を含む
        Checker.check_element_inclusion(
            "model_class", recipe, "model_class", "recipes of your input"
        )
        model_class = recipe["model_class"]

        # 確認2: model_class は _Model の派生クラス
        Checker.check_class_inheritance(model_class, _Model)

        # 確認3: model_class はアトリビュート _TUNABLE_PARAMS を含む
        Checker.check_attribute_existence(model_class, "_TUNABLE_PARAMS")

        # 確認4: model_class はアトリビュート _UNTUNABLE_PARAMS を含む
        Checker.check_attribute_existence(model_class, "_UNTUNABLE_PARAMS")

        return model_class

    @staticmethod
    def check_model_name(recipe: Dict[str, Any]) -> str:
        """
        確認: model_name は妥当

        Parameters
        ----------
        recipe : Dict[str, Any]
            モデルのレシピ

        Returns
        -------
        str
        """
        if "model_name" in recipe:
            # 確認: model_name は str
            Checker.check_param_type(recipe["model_name"], "model_name", str)
            return recipe["model_name"]
        else:
            return "no_name"

    @staticmethod
    def check_fixed_params(recipe: Dict[str, Any]) -> Dict[str, Any]:
        """
        確認: fixed_params は妥当

        Parameters
        ----------
        recipe : Dict[str, Any]
            モデルのレシピ

        Returns
        -------
        Dict[str, Any]
        """
        if "fixed_params" in recipe:
            fixed_params = recipe["fixed_params"]
            model_class = recipe["model_class"]

            # 確認1: fixed_params は Dict
            Checker.check_param_type(fixed_params, "fixed_params", Dict)

            # 処理: モデルのアトリビュート _TUNABLE_PARAMS と _UNTUNABLE_PARAMS のキーからなるリストをつくる
            input_params_list = list(fixed_params.keys())
            receivable_params_list = list(model_class._TUNABLE_PARAMS.keys()) + list(
                model_class._UNTUNABLE_PARAMS.keys()
            )

            # 確認2: model_class の _TUNABLE_PARAMS と _UNTUNABLE_PARAMS のキーは、 fixed_params の各キーを含む
            Checker.check_list_subset(
                input_params_list,
                receivable_params_list,
                "fixed-parameters of your input",
                "parameters the model can receive",
            )

            return fixed_params

        else:
            return {}

    @staticmethod
    def check_make_loss_func(
        recipe: Dict[str, Any],
        bunch_name: str,
        additions: Optional[List[str]] = None,
    ) -> Optional[Callable]:
        """
        確認: カスタム損失関数のエンクロージャは妥当

        Parameters
        ----------
        recipe : Dict[str, Any]
            モデルのレシピ

        bunch_name : str
            バンチの名前

        additions : Optional[List[str]]
            損失関数や評価関数に必要なカラム

        Returns
        -------
        Optional[Callable]
        """
        if "make_loss_func" in recipe:
            model_class = recipe["model_class"]
            make_loss_func = recipe["make_loss_func"]

            # 確認1: make_loss_func は Callable
            Checker.check_callable(make_loss_func, "make_loss_func")

            n_args = Checker.count_arguments(make_loss_func)

            # 確認2: make_loss_func の引数の数は 1 か 2
            if n_args == 1:
                # 確認3: make_loss_func は is_for_optuna という引数をもち、bool
                Checker.check_argument_type(make_loss_func, "is_for_optuna", bool)

                # 処理: クロージャをつくる
                loss_func_optuna = make_loss_func(True)
                loss_func_train = make_loss_func(False)

            elif n_args == 2:
                # 確認3: make_loss_func は is_for_optuna という引数をもち、bool
                Checker.check_argument_type(make_loss_func, "is_for_optuna", bool)

                # 確認4: make_loss_func は df という引数をもち、pd.DataFrame
                Checker.check_argument_type(make_loss_func, "df", pd.DataFrame)

                # 処理: クロージャをつくる
                df_tmp = DumperLoader.load_learn(bunch_name)[additions].head(5)
                loss_func_optuna = make_loss_func(True, df_tmp)
                loss_func_train = make_loss_func(False, df_tmp)

            else:
                raise TypeError(
                    f"Function {make_loss_func.__name__} should have one or two arguments."
                )

            # 確認5: クロージャは妥当
            Validator.check_loss_func_optuna(loss_func_optuna)
            model_class._check_loss_func(loss_func_train)

            return make_loss_func

        else:
            return None

    @staticmethod
    def check_loss_func_optuna(loss_func_optuna: Callable) -> None:
        """
        確認: optuna で最適なハイパーパラメータを探索するための損失関数は妥当

        Parameters
        ----------
        loss_func_optuna: Callable
            最適なハイパーパラメータを探索するための損失関数
        """
        # 確認1: loss_func_optuna は Callable
        Checker.check_callable(loss_func_optuna, loss_func_optuna.__name__)

        n_args = Checker.count_arguments(loss_func_optuna)

        # 確認2: loss_func_optuna の引数は2つ
        if n_args == 2:
            # 確認3: loss_func_optuna は preds という引数をもち、np.ndarray
            Checker.check_argument_type(loss_func_optuna, "preds", np.ndarray)

            # 確認4: loss_func_optuna は y という引数をもち、np.ndarray
            Checker.check_argument_type(loss_func_optuna, "y", np.ndarray)

            # 確認5: loss_func_optuna の戻り値は float
            Checker.check_return_type(loss_func_optuna, float)

        else:
            raise TypeError(
                f"Function {loss_func_optuna.__name__} should have two arguments."
            )

    @staticmethod
    def check_make_eval_func(
        recipe: Dict[str, Any],
        bunch_name: str,
        additions: Optional[List[str]] = None,
    ) -> Optional[Callable]:
        """
        確認: カスタム評価関数のエンクロージャは妥当

        Parameters
        ----------
        recipe : Dict[str, Any]
            モデルのレシピ

        bunch_name : str
            バンチの名前

        additions : Optional[List[str]]
            損失関数や評価関数に必要なカラム

        Returns
        -------
        Optional[Callable]
        """
        if "make_eval_func" in recipe:
            make_eval_func = recipe["make_eval_func"]

            # 確認1: make_eval_func は Callable
            Checker.check_callable(make_eval_func, "make_eval_func")

            n_args = Checker.count_arguments(make_eval_func)

            # 確認2: make_eval_func の引数の数は 0 か 1
            if n_args == 0:
                # 処理: クロージャをつくる
                eval_func = make_eval_func()

            elif n_args == 1:
                # 確認3: make_eval_func は df という引数をもち、pd.DataFrame
                Checker.check_argument_type(make_eval_func, "df", pd.DataFrame)

                # 処理: クロージャをつくる
                df_tmp = DumperLoader.load_learn(bunch_name)[additions].head(5)
                eval_func = make_eval_func(df_tmp)

            else:
                raise TypeError(
                    f"Function {make_eval_func.__name__} should have one or no argument."
                )

            # 確認4: クロージャの型は妥当
            recipe["model_class"]._check_eval_func(eval_func)

            return make_eval_func

        else:
            return None

    @staticmethod
    def build_models(
        bunch_name: str,
        task: str,
        recipes: List[Dict[str, Any]],
    ) -> List[_Model]:
        """
        仕様: recipes にもとづいて、モデルリストを作る

        Parameters
        ----------
        bunch_name:
            バンチの名前

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
                    bunch_name,
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
            バンチの説明
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

    @staticmethod
    def check_n_trials(n_trials: int):
        """
        確認: n_trials は妥当

        Parameters
        ----------
        n_trials : int
            Optuna のハイパーパラメータの試行回数
        """
        # 確認: n_fold は自然数
        Checker.check_natural_number(n_trials, "n_trials")

    @staticmethod
    def check_random_seed(random_seed: int):
        """
        確認: random_seed は妥当

        Parameters
        ----------
        random_seed : int
            乱数生成のシード
        """
        # 確認: n_fold は自然数
        Checker.check_natural_number(random_seed, "random_seed")

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

    @property
    def is_available(self):
        return self._is_available
