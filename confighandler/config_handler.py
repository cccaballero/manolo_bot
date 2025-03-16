import json
import logging
import os
from typing import Any


class BaseField:
    def __init__(
        self,
        name: str,
        required: bool = False,
        default: str = None,
        error: str = None,
        lazy: bool = True,
        warning: str = None,
    ):
        self.name = name
        self.required = required
        self.default = default
        self.error = f"Environment variable {name} is required" if error is None else error
        self.lazy = lazy
        self.warning = warning

        self._config_value = None

    def get_env_variable(self) -> str:
        if not self.lazy and self._config_value is not None:
            return self._config_value
        self._config_value = os.getenv(self.name, self.default)
        return self._config_value

    def __call__(self) -> str:
        env_value = self.get_env_variable()
        if self.required and env_value is None:
            raise Exception(self.error)
        if self.warning and env_value is None:
            logging.warning(self.warning)
        return env_value


class StringField(BaseField):
    pass


class BooleanField(BaseField):
    def __init__(self, name, default: bool = False, **kwargs):
        super().__init__(name, **kwargs)
        self.default = default

    def get_env_variable(self) -> str:
        if not self.lazy and self._config_value is not None:
            return self._config_value
        env_value = os.getenv(self.name, None)
        if env_value:
            self._config_value = env_value.lower() in ("true", "1", "t")
        else:
            self._config_value = self.default
        return self._config_value


class IntegerField(BaseField):
    def __init__(self, name, default: int = None, **kwargs):
        super().__init__(name, **kwargs)
        self.default = default

    def get_env_variable(self) -> str:
        if not self.lazy and self._config_value is not None:
            return self._config_value
        try:
            self._config_value = int(os.getenv(self.name, self.default))
        except ValueError:
            raise Exception(f"Environment variable {self.name} must be an integer")
        return self._config_value


class FloatField(BaseField):
    def __init__(self, name, default: float = None, **kwargs):
        super().__init__(name, **kwargs)
        self.default = default

    def get_env_variable(self) -> str:
        if not self.lazy and self._config_value is not None:
            return self._config_value
        try:
            self._config_value = float(os.getenv(self.name, self.default))
        except ValueError:
            raise Exception(f"Environment variable {self.name} must be a float")
        return self._config_value


class JsonField(BaseField):
    def __init__(self, name, default: dict = None, **kwargs):
        super().__init__(name, **kwargs)
        self.default = default

    def get_env_variable(self) -> str:
        if not self.lazy and self._config_value is not None:
            return self._config_value
        try:
            env_value = os.getenv(self.name, None)
            if env_value:
                self._config_value = json.loads(env_value)
            else:
                self._config_value = self.default
                if self.warning:
                    logging.warning(self.warning)
        except Exception as e:
            logging.exception(e)
            self._config_value = self.default
            if self.warning:
                logging.warning(self.warning)
        return self._config_value


class StringListField(BaseField):
    def __init__(self, name, default: list[str] = None, **kwargs):
        super().__init__(name, **kwargs)
        self.default = default

    def get_env_variable(self) -> str:
        if not self.lazy and self._config_value is not None:
            return self._config_value
        try:
            env_value = os.getenv(self.name, None)
            if env_value:
                self._config_value = [element.strip() for element in env_value.split(",") if element.strip()]
            else:
                self._config_value = self.default
        except Exception:
            raise Exception(f"Environment variable {self.name} must be a valid (comma separated) list of strings")
        return self._config_value


class BaseConfig:
    def __init__(self, lazy=False):
        self.lazy = lazy
        if not self.lazy:
            self._touch_all_attributes()

    def _touch_all_attributes(self):
        for name, field in self.__class__.__dict__.items():
            if isinstance(field, BaseField):
                field()

    def __getattribute__(self, item: Any) -> Any:
        if isinstance(object.__getattribute__(self, item), BaseField):
            return object.__getattribute__(self, item)()
        return object.__getattribute__(self, item)
