import logging

_NOISY_LOGGERS = (
    "asyncio",
    "deepagents",
    "httpcore",
    "httpcore.http11",
    "httpx",
    "langsmith",
    "mcp",
    "urllib3",
)

_ERROR_LOGGERS = ("langchain_google_genai",)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
        level=level,
        force=True,
    )
    for logger_name in _NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    for logger_name in _ERROR_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
