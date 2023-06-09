import datetime
import logging
from pathlib import Path
import pandas as pd

from mypoc.config.config import config


class Logger:
    """
    ログ出力を行うクラス。

    このクラスは一般的な情報（info）と実行結果（result）をログとして記録します。
    それぞれについて、ストリーム（通常はコンソール）とファイルへのハンドラーが設定されています。
    また、計算結果については、辞書形式で LTSV 形式（ラベル-タブ区切り値）に変換し、出力します。

    Attributes
    ----------
    general_logger : Logger
        一般的な情報を出力するためのロガー。
    result_logger : Logger
        実行結果を出力するためのロガー。
    """

    def __init__(
        self,
        path_general: Path = Path(config["PATH_GENERAL"]),
        path_result: Path = Path(config["PATH_RESULT"]),
    ):
        # 2つのロガーを作成
        self.general_logger = logging.getLogger("general")
        self.result_logger = logging.getLogger("result")

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # ファイルハンドラーを作成
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        file_general_handler = logging.FileHandler(path_general)
        file_general_handler.setFormatter(formatter)

        file_result_handler = logging.FileHandler(path_result)
        file_result_handler.setFormatter(formatter)

        # 同じハンドラーを複数回追加して、ログメッセージを重複して出力するのを防ぐため
        if len(self.general_logger.handlers) == 0:
            self.general_logger.addHandler(stream_handler)
            self.general_logger.addHandler(file_general_handler)
            self.result_logger.addHandler(stream_handler)
            self.result_logger.addHandler(file_result_handler)

            # INFO レベル以上（INFO, WARNING, ERROR, CRITICAL）のログメッセージを出力
            self.general_logger.setLevel(logging.INFO)
            self.result_logger.setLevel(logging.INFO)

    def info(self, message: str):
        """
        仕様: 現時刻を付加して、情報をロガーに出力

        Parameters
        ----------
        message : str
            出力するメッセージ
        """
        self.general_logger.info("[{}] - {}".format(self.now_string(), message))

    def result(self, message: str):
        """
        仕様: 実行結果をロガーに出力

        Parameters
        ----------
        message : str
            出力するメッセージ
        """
        self.result_logger.info(message)

    def result_scores(self, df: pd.DataFrame) -> None:
        """
        仕様: 指定のデータフレームを LTSV 形式でロガーに出力

        Parameters
        ----------
        df : pd.DataFrame
            指定のデータフレーム
        """
        for _, row in df.iterrows():
            ltsv_line = "\t".join(
                f"{key}:{value}" for key, value in row.to_dict().items()
            )
            self.result(ltsv_line)

    def now_string(self):
        return str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
