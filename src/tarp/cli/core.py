from tarp.cli.logging.color import ColoredLogger
from tarp.cli.logging.core import Logger


class Console:
    _backend: Logger | None = None

    @classmethod
    def _get_backend(cls) -> Logger:
        """Lazy loader fallback to resolve circular dependencies cleanly."""
        if cls._backend is None:
            cls._backend = ColoredLogger()
        return cls._backend

    @classmethod
    def use(cls, backend: Logger) -> None:
        """Replace the operational backend configuration at runtime safely."""
        cls._backend = backend

    @classmethod
    def debug(cls, message: str) -> None:
        cls._get_backend().debug(message)

    @classmethod
    def info(cls, message: str) -> None:
        cls._get_backend().info(message)

    @classmethod
    def warning(cls, message: str) -> None:
        cls._get_backend().warning(message)

    @classmethod
    def error(cls, message: str) -> None:
        cls._get_backend().error(message)

    @classmethod
    def critical(cls, message: str) -> None:
        cls._get_backend().critical(message)

    @classmethod
    def exception(cls, message: str, exc_info: bool = True) -> None:
        cls._get_backend().exception(message, exc_info=exc_info)
