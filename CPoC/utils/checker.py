# 標準ライブラリ
from inspect import signature
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union


class Checker:
    @staticmethod
    def check_param_type(
        param: Any,
        name: str,
        expected_type: Type[Any],
    ) -> None:
        """
        確認: 変数が期待どおりの組込みデータ型

        Parameters
        ----------
        param : Any
            変数

        name : str
            変数の名前 (エラーメッセージで使用)

        expected_type : type
            期待する組込みデータ型
        """
        if not isinstance(param, expected_type):
            raise TypeError(f"Parameter '{name}' must be of type {expected_type}.")

    @staticmethod
    def check_callable(
        param: Any,
        name: str,
    ) -> None:
        """
        確認: 変数が呼び出し可能

        Parameters
        ----------
        param : Any
            変数

        name : str
            変数の名前 (エラーメッセージで使用)
        """
        if not callable(param):
            raise TypeError(f"Parameter '{name}' must be of type Callable.")

    @staticmethod
    def check_list_elements_type(
        list_obj: List[Any],
        list_name: str,
        expected_type: Type[Any],
    ) -> None:
        """
        確認: リストの各要素は期待どおりの組込みデータ型

        Parameters
        ----------
        list_obj : list
            リスト

        list_name : str
            リストの名前 (エラーメッセージで使用)

        expected_type : type
            期待するデータ型
        """
        # 確認1: list_obj は List
        Checker.check_param_type(list_obj, list_name, List)

        # 確認2: list_obj の各要素は expected_type
        for i, elem in enumerate(list_obj):
            if not isinstance(elem, expected_type):
                raise TypeError(
                    f"Each element of '{list_name}' must be of type {expected_type}. But the element at index {i} is {type(elem)}."
                )

    @staticmethod
    def check_no_duplication(
        list_obj: List,
        list_name: str,
    ):
        """
        確認: 指定のリストに重複はない

        Parameters
        ----------
        list_obj : List
            指定のリスト

        list_name : str
            リストの名前 (エラーメッセージで使用)
        """
        if len(list_obj) != len(set(list_obj)):
            raise ValueError(f"{list_name} contains duplicate elements.")

    @staticmethod
    def check_value_no_duplication(
        dict_list: List[Dict[str, Any]],
        dict_list_name: str,
        key: str,
    ) -> None:
        """
        確認: 辞書のリストの指定したキーの値は重複しない

        Parameters
        ----------
        dict_list : List[Dict]
            指定した辞書のリスト

        dict_list_name : str
            指定した辞書のリストの名前

        key : str
            重複を確認するキー
        """
        key_values = []
        for i, d in enumerate(dict_list):
            Checker.check_element_inclusion(
                key, d, key, f"the dictionary at index {i} in {dict_list_name}"
            )
            key_values.append(d[key])

        Checker.check_no_duplication(key_values, f"values of {key} in {dict_list_name}")

    @staticmethod
    def check_element_inclusion(
        item: Any,
        container: Any,
        item_name: Optional[str] = None,
        container_name: Optional[str] = None,
    ) -> None:
        """
        確認: ある要素が他のオブジェクトに含まれている

        Parameters
        ----------
        item : Any
            含まれるべき要素

        container : Any
            含むべきオブジェクト

        item_name: Optional[str]
            含まれるべき要素の名称 (エラーメッセージで使用)

        container_name: Optional[str]
            含むべきオブジェクトの名称 (エラーメッセージで使用)
        """
        if item not in container:
            item_name = item_name if item_name else item.__class__.__name__
            container_name = (
                container_name if container_name else container.__class__.__name__
            )
            raise ValueError(f"The {item_name} is not included in {container_name}.")

    @staticmethod
    def check_element_exclusion(
        item: Any,
        container: Any,
        item_name: Optional[str] = None,
        container_name: Optional[str] = None,
    ) -> None:
        """
        確認: ある要素が他のオブジェクトに含まれていない

        Parameters
        ----------
        item : Any
            含まれるべきではない要素

        container : Any
            含むべきではないオブジェクト

        item_name: Optional[str]
            含まれるべきではない要素の名称 (エラーメッセージで使用)

        container_name: Optional[str]
            含むべきではないオブジェクトの名称 (エラーメッセージで使用)
        """
        if item in container:
            item_name = item_name if item_name else item.__class__.__name__
            container_name = (
                container_name if container_name else container.__class__.__name__
            )
            raise ValueError(f"The {item_name} is included in {container_name}.")

    @staticmethod
    def check_list_subset(
        list1: List[Any],
        list2: List[Any],
        list1_name: str,
        list2_name: str,
    ) -> None:
        """
        確認: あるリストの全要素を片方のリストが含む

        Parameters
        ----------
        list1 : List[Any]
            含まれるべきリスト

        list2 : List[Any]
            含むべきリスト

        list1_name: str
            含まれるべきリストの名前 (エラーメッセージで使用)

        list2_name: str
            含むべきリストの名前 (エラーメッセージで使用)
        """
        set1 = set(list1)
        set2 = set(list2)

        if not set1.issubset(set2):
            missing_elements = set1.difference(set2)
            raise ValueError(
                f"The elements {missing_elements} from {list1_name} are not included in {list2_name}."
            )

    @staticmethod
    def check_path_existence(path: Path):
        """
        確認: パスが有効

        Parameters
        ----------
        path : Path
            パス
        """
        if not path.exists():
            raise FileNotFoundError(f"The '{str(path)}' is invalid.")

    @staticmethod
    def check_path_no_existence(path: Path):
        """
        確認: パスがない

        Parameters
        ----------
        path : Path
            パス
        """
        if path.exists():
            raise FileNotFoundError(f"The '{str(path)}' already exists.")

    @staticmethod
    def check_attribute_existence(
        obj: Union[Type[Any], Any],
        attribute_name: str,
    ) -> None:
        """
        確認: クラスまたはインスタンスは指定のアトリビュートをもつ

        Parameters
        ----------
        obj : Union[Type[Any], Any]
            クラスまたはインスタンス

        attribute_name : str
            アトリビュートの名前
        """
        if not hasattr(obj, attribute_name):
            raise AttributeError(
                f"The object '{obj.__class__.__name__ if not isinstance(obj, type) else obj.__name__}' does not have attribute '{attribute_name}'."
            )

    @staticmethod
    def check_class_inheritance(
        obj: Union[Type, Any],
        baseclass: Type,
    ) -> None:
        """
        確認: クラスまたはインスタンスが指定のクラスを継承

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
    def check_natural_number(
        value: int,
        value_name: str,
    ) -> None:
        """
        確認: 変数は自然数

        Parameters
        ----------
        value : int
            変数

        value_name : str
            変数の名前 (エラーメッセージで使う)
        """
        # 確認1:
        Checker.check_param_type(value, f"{value_name}", int)

        if value < 1:
            raise ValueError(
                f"'{value_name}' should be a natural number (greater than 0), but got {value}."
            )

    @staticmethod
    def check_existence(
        obj: Any,
        obj_name: Optional[str] = None,
    ) -> None:
        """
        確認: 指定の変数が、要素を含んでいる\n
        例: 空文字だとエラー

        Parameters
        ----------
        obj : Any
            変数

        obj_name : Optional[str]
            変数の名前 (エラーメッセージで使う)\n
            指定しない場合、その変数のクラス名を使う
        """
        obj_name = obj_name or obj.__class__.__name__ if obj else "Object"
        if not obj:
            raise ValueError(f"The {obj_name} should not be empty.")

    @staticmethod
    def count_arguments(func):
        """
        関数の引数の数を数える

        Parameters
        ----------
        func : Callable
            関数

        Returns
        -------
        int
            関数の引数の数
        """
        sig = signature(func)
        params = sig.parameters

        return len(params)

    @staticmethod
    def check_argument_type(
        func: Callable,
        arg_name: str,
        arg_type: type,
    ) -> None:
        """
        確認: 関数が指定の引数を指定のデータ型でもつ

        Parameters
        ----------
        func : Callable
            関数

        arg_name : str
            指定の引数の名前

        arg_type : type
            指定のデータ型
        """
        sig = signature(func)
        params = sig.parameters

        # 確認1: 関数は指定の引数を含む
        if arg_name not in params:
            raise ValueError(
                f"Function '{func.__name__}' does not have argument '{arg_name}'"
            )

        # 確認2: 引数は指定のデータ型
        arg_annotation = params[arg_name].annotation
        if arg_annotation != arg_type:
            raise ValueError(
                f"Argument '{arg_name}' in function '{func.__name__}' is not of type '{arg_type.__name__}', it's of type '{arg_annotation.__name__}'"
            )

    @staticmethod
    def check_return_type(
        func: Callable,
        expected_type: Type,
    ) -> None:
        """
        確認: 関数の戻り値は指定の戻り値

        Parameters
        ----------
        func : Callable
            関数

        expected_type : Type
            指定の戻り値
        """
        sig = signature(func)
        return_annotation = sig.return_annotation

        # 確認1: 戻り値を型注釈している
        if return_annotation is not sig.empty:
            # 確認2: 関数の戻り値は指定の戻り値
            if return_annotation is not expected_type:
                raise TypeError(
                    f"Function {func.__name__} should return a value of type {expected_type.__name__}, "
                    f"but returns a value of type {return_annotation.__name__}."
                )
        else:
            raise TypeError(
                f"Function {func.__name__} has no return type annotation. "
                f"It should return a value of type {expected_type.__name__}."
            )
