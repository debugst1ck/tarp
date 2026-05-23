import logging
from pathlib import Path
from typing import override

from tqdm.auto import tqdm

from tarp.cli.logging.core import Logger


class TqdmLoggingHandler(logging.Handler):
    """Thread-safe context logging handler that writes without breaking tqdm progress bars."""

    @override
    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            tqdm.write(message)
        except Exception:
            self.handleError(record)


class ColorFormatterANSI(logging.Formatter):
    """Applies ANSI colors strictly to terminal output streams."""

    _COLORS: dict[str, str] = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARN": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    _RESET: str = "\033[0m"

    @override
    def format(self, record: logging.LogRecord) -> str:
        color = self._COLORS.get(record.levelname, self._RESET)
        # Minimize string construction overhead
        return f"{color}[{record.levelname}]\t{self.formatTime(record, self.datefmt)} >>> {record.getMessage()}{self._RESET}"


class ColoredLogger(Logger):
    """A logger that outputs colored log messages using tqdm's thread-safe write method."""

    def __init__(
        self, directory: str | Path = "logs", name: str = "tarp_logger"
    ) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        file_path = directory / f"{name}.log"

        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False

        if self._logger.handlers:
            for handler in self._logger.handlers:
                self._logger.removeHandler(handler)

        logging.addLevelName(logging.WARNING, "WARN")

        console_handler = TqdmLoggingHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(ColorFormatterANSI())
        self._logger.addHandler(console_handler)

        file_handler = logging.FileHandler(
            file_path.as_posix(), mode="a", encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("[%(levelname)s]\t%(asctime)s >>> %(message)s")
        )
        self._logger.addHandler(file_handler)

    @override
    def debug(self, message: str) -> None:
        self._logger.debug(message)

    @override
    def info(self, message: str) -> None:
        self._logger.info(message)

    @override
    def warning(self, message: str) -> None:
        self._logger.warning(message)

    @override
    def error(self, message: str) -> None:
        self._logger.error(message)

    @override
    def critical(self, message: str) -> None:
        self._logger.critical(message)

    @override
    def exception(self, message: str, exc_info: bool = True) -> None:
        self._logger.exception(message, exc_info=exc_info)
