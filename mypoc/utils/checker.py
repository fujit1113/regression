# 標準
from inspect import signature
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Type, Union


class Checker:
    @staticmethod
    def _generate_error_message(
        name: str,
        expected: Union[type, str],
    ) -> str:
        """
        仕様: エラーメッセージを生成

        Parameters
        ----------
        name : str
            パラメータの名前

        expected_type : type
            期待するデータ型

        Returns
        -------
        str
            生成されたエラーメッセージ
        """
        if isinstance(expected, type):
            return f"Parameter '{name}' must be of type {expected.__name__}."
        else:
            return f"Parameter '{name}' must be {expected}."

    @staticmethod
    def check_param_type(
        param: Any,
        name: str,
        expected_type: Type[Any],
    ) -> None:
        """
        確認: 指定の変数が期待どおりの組込みデータ型

        Parameters
        ----------
        param : Any
            指定の変数

        name : str
            変数の名前 (エラーメッセージで使用)

        expected_type : type
            期待する組込みデータ型
        """
        if not isinstance(param, expected_type):
            raise TypeError(Checker._generate_error_message(name, expected_type))

    @staticmethod
    def check_callable(
        param: Any,
        name: str,
    ) -> None:
        """
        確認: 指定の変数が呼び出し可能

        Parameters
        ----------
        param : Any
            指定の変数

        name : str
            変数の名前 (エラーメッセージで使用)
        """
        if not callable(param):
            raise TypeError(Checker._generate_error_message(name, "callable"))

    @staticmethod
    def check_list_elements_type(
        list_obj: List[Any],
        list_name: str,
        expected_type: Type[Any],
    ) -> None:
        """
        確認: 指定のリストの各要素は期待どおりの組込みデータ型

        Parameters
        ----------
        list_obj : list
            指定のリスト

        list_name : str
            リストの名前 (エラーメッセージで使用)

        expected_type : type
            期待するデータ型
        """
        # 確認1: list_obj は List 型
        Checker.check_param_type(list_obj, list_name, List)

        # 確認2: list_obj の各要素は expected_type 型
        for i, elem in enumerate(list_obj):
            if not isinstance(elem, expected_type):
                raise TypeError(
                    f"Each element of '{list_name}' must be of type {expected_type}. But the element at index {i} is {type(elem)}."
                )

    @staticmethod
    def check_duplicate_keys(
        dict_list: List[Dict[str, Any]],
        key: str,
    ) -> None:
        """
        確認: 指定の辞書のリストに指定したキーの値に重複がない

        Parameters
        ----------
        dict_list : List[Dict]
            指定した辞書のリスト

        key : str
            重複を確認するキー
        """
        key_values = []
        for i, d in enumerate(dict_list):
            if key not in d:
                raise KeyError(
                    f"The required key '{key}' is missing in the dictionary at index {i} in the list."
                )
            key_values.append(d[key])

        if len(key_values) != len(set(key_values)):
            raise ValueError(
                f"Duplicate values found for key '{key}' in the dictionary list."
            )

    @staticmethod
    def check_list_inclusion(
        list1: List[Any],
        list1_name: str,
        list2: List[Any],
        list2_name: str,
    ) -> None:
        """
        確認: あるリストの全要素を片方のリストが含む

        Parameters
        ----------
        list1 : List[Any]
            含まれるべきリスト

        list1_name: str
            含まれるべきリストの名称 (エラーメッセージで使用)

        list2 : List[Any]
            含むべきリスト

        list2_name: str
            含むべきリストの名称 (エラーメッセージで使用)
        """
        set1 = set(list1)
        set2 = set(list2)

        if not set1.issubset(set2):
            missing_elements = set1.difference(set2)
            raise ValueError(
                f"The elements {missing_elements} from {list1_name} are not included in {list2_name}."
            )

    @staticmethod
    def check_no_common_elements(
        list1: List[Any],
        list1_name: str,
        list2: List[Any],
        list2_name: str,
    ) -> None:
        """
        確認: リスト間で共通する要素がない

        Parameters
        ----------
        list1 : List[Any]
            リスト1

        list1_name : str
            リスト1の名称 (エラーメッセージで使用)

        list2 : List[Any]
            リスト2

        list2_name : str
            リスト2の名称 (エラーメッセージで使用)
        """
        set1 = set(list1)
        set2 = set(list2)

        common_elements = set1.intersection(set2)
        if common_elements:
            raise ValueError(
                f"{list1_name} and {list2_name} have common elements: {common_elements}"
            )

    @staticmethod
    def check_path_exists(path: Path):
        """
        確認: 指定のパスが存在

        Parameters
        ----------
        path : str
            指定のパス
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"The '{path}' is invalid.")

    @staticmethod
    def check_no_path_exists(path: Path):
        """
        確認: 指定のパスが存在しない

        Parameters
        ----------
        path : str
            指定のパス
        """
        if os.path.exists(path):
            raise FileNotFoundError(f"The '{path}' already exists.")

    @staticmethod
    def check_has_attribute(
        obj: Union[Type[Any], Any],
        attribute_name: str,
    ) -> None:
        """
        確認: 指定のクラスまたはインスタンスは指定のアトリビュートを持つ

        Parameters
        ----------
        obj : Union[Type[Any], Any]
            指定のクラスまたはインスタンス

        attribute_name : str
            アトリビュートの名前
        """
        if not hasattr(obj, attribute_name):
            raise AttributeError(
                f"The object '{obj.__class__.__name__ if not isinstance(obj, type) else obj.__name__}' does not have attribute '{attribute_name}'."
            )

    @staticmethod
    def check_inheritance(
        obj: Union[Type, Any],
        baseclass: Type,
    ) -> None:
        """
        確認: 指定のクラスまたはインスタンスが指定のクラスを継承

        Parameters
        ----------
        obj : Union[Type, Any]
            派生クラスまたはそのインスタンス

        baseclass : Type
            基底クラス
        """
        # 　仕様1: obj がクラス
        if isinstance(obj, type):
            if not issubclass(obj, baseclass):
                raise TypeError(
                    f"{obj.__name__} does not inherit from {baseclass.__name__}"
                )

        # 　仕様2: obj がインスタンス
        else:
            if not isinstance(obj, baseclass):
                raise TypeError(
                    f"The instance is not derived from {baseclass.__name__}"
                )

    @staticmethod
    def check_key_in_dict(
        key: str,
        dictionary: Dict[str, Any],
    ) -> None:
        """
        確認: 指定の辞書は指定のキーを含む

        Parameters
        ----------
        key: str
            指定のキー

        dictionary: Type
            指定の辞書
        """
        if key not in dictionary:
            raise KeyError(f"The key '{key}' is not in the dictionary.")

    @staticmethod
    def check_func_args(
        func: Callable,
        max_args: int,
    ) -> None:
        """
        確認: 指定の関数の引数の数は、指定の数未満

        Parameters
        ----------
        func : Callable
            指定の関数

        max_args : int
            指定の引数の数
        """
        sig = signature(func)
        params = sig.parameters

        if len(params) > max_args:
            raise TypeError(
                f"Function {func.__name__} should have {max_args} or less arguments."
            )

    @staticmethod
    def check_natural_number(
        value: int,
        variable_name: str,
    ) -> None:
        """
        確認: 指定の変数は自然数

        Parameters
        ----------
        value : int
            指定の変数

        variable_name : str
            指定の変数名
        """
        # 確認1:
        Checker.check_param_type(value, f"{variable_name}", int)

        if value < 1:
            raise ValueError(
                f"'{variable_name}' should be a natural number (greater than 0), but got {value}."
            )
