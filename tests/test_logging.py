import logging
import unittest
from unittest.mock import patch

from manolo_bot.logging_config import _ERROR_LOGGERS, _NOISY_LOGGERS, configure_logging


class TestLoggingConfiguration(unittest.TestCase):
    def test_noisy_loggers_are_capped_at_warning(self):
        root_logger = logging.getLogger()
        original_root_level = root_logger.level
        original_logger_levels = {name: logging.getLogger(name).level for name in (*_NOISY_LOGGERS, *_ERROR_LOGGERS)}

        def configure_root_logger(**kwargs):
            root_logger.setLevel(kwargs["level"])

        try:
            with patch(
                "manolo_bot.logging_config.logging.basicConfig", side_effect=configure_root_logger
            ) as basic_config:
                configure_logging("DEBUG")

            basic_config.assert_called_once_with(
                format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
                level="DEBUG",
                force=True,
            )
            self.assertEqual(root_logger.level, logging.DEBUG)
            self.assertEqual(logging.getLogger("manolo_bot").getEffectiveLevel(), logging.DEBUG)
            for logger_name in _NOISY_LOGGERS:
                with self.subTest(logger=logger_name):
                    self.assertEqual(logging.getLogger(logger_name).level, logging.WARNING)
            for logger_name in _ERROR_LOGGERS:
                with self.subTest(logger=logger_name):
                    self.assertEqual(logging.getLogger(logger_name).level, logging.ERROR)
        finally:
            root_logger.setLevel(original_root_level)
            for logger_name, level in original_logger_levels.items():
                logging.getLogger(logger_name).setLevel(level)
