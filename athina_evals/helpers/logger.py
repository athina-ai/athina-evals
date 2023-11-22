import logging
import colorlog
from typing import Dict, Any


class Singleton(type):
    _instances: Dict[Any, Any] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class AppLogger(logging.Logger, metaclass=Singleton):
    """
    Custom logger class that supports color and file logging.
    """

    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)

        # Create a console handler with color support
        console_handler = colorlog.StreamHandler()
        console_handler.setFormatter(
            colorlog.ColoredFormatter(
                "%(log_color)s%(message)s",
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "white",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                },
            )
        )
        self.addHandler(console_handler)

    def args_str(self, *args):
        return ", ".join([str(arg) for arg in args])

    def debug(self, message, *args):
        args_str = self.args_str(*args)
        super(AppLogger, self).debug(f"{message}\n{args_str}")

    def info(self, message, *args):
        args_str = self.args_str(*args)
        super(AppLogger, self).info(f"{message}\n{args_str}")

    def success(self, message, *args):
        args_str = self.args_str(*args)
        # Call the base class's info method to prevent recursion
        super(AppLogger, self).info(f"\033[32m{message}\n{args_str}\033[0m")

    def error(self, message, *args):
        args_str = self.args_str(*args)
        super(AppLogger, self).error("ERROR: " + message + "\n" + args_str)

    def warning(self, message, *args):
        args_str = self.args_str(*args)
        super(AppLogger, self).warning("WARN: " + message + "\n" + args_str)

    def log_with_color(self, level, message, color, *args, **kwargs):
        colors = {
            "black": "30",
            "red": "31",
            "green": "32",
            "yellow": "33",
            "blue": "34",
            "magenta": "35",
            "cyan": "36",
            "white": "37",
        }

        color_code = colors.get(color.lower(), "37")
        formatted_message = f"\033[{color_code}m{message}\033[0m"
        self._log(level, formatted_message, args)

    def to_file(self, output: str, log_file):
        if log_file is not None:
            log_file.write(output + "\n")
            log_file.flush()  # Ensure immediate writing to the file

    def to_file_and_console(self, output: str, log_file=None, color=None):
        self.to_file(output, log_file)

        if color is not None:
            logger.log_with_color(output, color)
        else:
            logger.info(output)


def setup_logger():
    logger = AppLogger("app_logger", level=logging.DEBUG)
    return logger


# Create a default logger instance
logger = setup_logger()
