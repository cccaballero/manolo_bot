import os
import unittest
from unittest.mock import patch

from confighandler import (
    BaseConfig,
    BooleanField,
    FloatField,
    IntegerField,
    JsonField,
    StringField,
    StringListField,
)


class Cfg(BaseConfig):
    a = StringField("A_TEST_ENV_VAR", required=False)


class Cfg2(BaseConfig):
    a = StringField("A_TEST_ENV_VAR2", required=True)


class TestConfig(unittest.TestCase):
    def test_string_field__inherits_base_field_behavior(self):
        # Arrange
        import os
        from unittest.mock import patch

        test_name = "TEST_VAR"
        test_value = "test_value"

        # Act
        with patch.dict(os.environ, {test_name: test_value}):
            string_field = StringField(test_name)
            result = string_field()

        # Assert
        self.assertEqual(result, test_value)
        self.assertEqual(string_field.name, test_name)
        self.assertFalse(string_field.required)
        self.assertIsNone(string_field.default)

    def test_string_field__required_raises_exception(self):
        # Arrange
        test_name = "NONEXISTENT_VAR"
        custom_error = "Custom error message"

        # Act & Assert
        with patch.dict(os.environ, {}, clear=True):
            string_field = StringField(test_name, required=True, error=custom_error)
            with self.assertRaises(Exception) as context:
                string_field()

            self.assertEqual(str(context.exception), custom_error)

    def test_boolean_field__returns_true_when_env_var_is_true(self):
        # Arrange
        test_env_var = "TEST_BOOL_VAR"

        # Act
        with patch.dict(os.environ, {test_env_var: "true"}):
            boolean_field = BooleanField(test_env_var)
            result = boolean_field.get_env_variable()

        # Assert
        self.assertTrue(result)
        self.assertTrue(boolean_field._config_value)

    def test_boolean_field__handles_case_insensitive_boolean_values(self):
        # Arrange
        test_env_var = "TEST_BOOL_VAR"
        test_values = ["TRUE", "True", "true", "T", "t", "1"]

        # Act & Assert
        for value in test_values:
            with patch.dict(os.environ, {test_env_var: value}):
                boolean_field = BooleanField(test_env_var)
                result = boolean_field.get_env_variable()
                self.assertTrue(result, f"Failed for value: {value}")

    def test_integer_field__retrieves_and_converts_valid_integer(self):
        # Arrange
        os.environ["TEST_INT"] = "42"
        integer_field = IntegerField("TEST_INT")

        # Act
        result = integer_field.get_env_variable()

        # Assert
        self.assertEqual(result, 42)
        self.assertEqual(integer_field._config_value, 42)

        # Cleanup
        del os.environ["TEST_INT"]

    def test_integer_field__raises_exception_for_invalid_integer(self):
        # Arrange
        os.environ["TEST_INT"] = "not_an_integer"
        integer_field = IntegerField("TEST_INT")

        # Act & Assert
        with self.assertRaises(Exception) as context:
            integer_field.get_env_variable()

        self.assertEqual(str(context.exception), "Environment variable TEST_INT must be an integer")

        # Cleanup
        del os.environ["TEST_INT"]

    def test_float_field__get_valid_float_env_variable(self):
        # Arrange
        test_var_name = "TEST_FLOAT_VAR"
        test_float_value = "3.14"
        os.environ[test_var_name] = test_float_value
        float_field = FloatField(test_var_name)

        # Act
        result = float_field.get_env_variable()

        # Assert
        self.assertEqual(result, 3.14)
        self.assertIsInstance(result, float)

        # Cleanup
        del os.environ[test_var_name]

    def test_float_field__raises_exception_for_non_float_value(self):
        # Arrange
        test_var_name = "TEST_FLOAT_VAR"
        test_invalid_value = "not_a_float"
        os.environ[test_var_name] = test_invalid_value
        float_field = FloatField(test_var_name)

        # Act & Assert
        with self.assertRaises(Exception) as context:
            float_field.get_env_variable()

        self.assertEqual(str(context.exception), f"Environment variable {test_var_name} must be a float")

        # Cleanup
        del os.environ[test_var_name]

    def test_json_field__valid_json_is_correctly_parsed(self):
        # Arrange
        test_name = "TEST_JSON_ENV"
        test_json = '{"key": "value", "number": 42}'
        expected_dict = {"key": "value", "number": 42}

        # Set environment variable
        with patch.dict(os.environ, {test_name: test_json}):
            # Act
            json_field = JsonField(test_name)
            result = json_field()

            # Assert
            self.assertEqual(result, expected_dict)
            self.assertEqual(json_field._config_value, expected_dict)

    def test_json_field__invalid_json_falls_back_to_default(self):
        # Arrange
        test_name = "TEST_INVALID_JSON_ENV"
        invalid_json = "{not valid json}"
        default_value = {"default": "value"}
        warning_message = "Custom warning message"

        # Set environment variable
        with patch.dict(os.environ, {test_name: invalid_json}):
            # Act
            with self.assertLogs(level="WARNING") as log_context:
                with patch("logging.exception") as mock_log_exception:
                    json_field = JsonField(test_name, default=default_value, warning=warning_message)
                    result = json_field()

            # Assert
            self.assertEqual(result, default_value)
            self.assertEqual(json_field._config_value, default_value)
            mock_log_exception.assert_called_once()
            self.assertIn(warning_message, log_context.output[0])

    def test_string_list_field__parses_comma_separated_string(self):
        # Arrange
        field = StringListField("TEST_LIST_VAR")
        with patch.dict(os.environ, {"TEST_LIST_VAR": "item1, item2,item3, item4 "}):
            # Act
            result = field()

            # Assert
            self.assertEqual(result, ["item1", "item2", "item3", "item4"])

    def test_string_list_field__handles_single_value(self):
        # Arrange
        field = StringListField("TEST_SINGLE_VAR")
        with patch.dict(os.environ, {"TEST_SINGLE_VAR": "single_item"}):
            # Act
            result = field()

            # Assert
            self.assertEqual(result, ["single_item"])
            self.assertIsInstance(result, list)

    def test_basic_config(self):
        os.environ["A_TEST_ENV_VAR"] = "a test variable"
        cfg = Cfg()
        self.assertEqual(cfg.a, "a test variable")

    def test_basic_config__required(self):
        cfg = Cfg2(lazy=True)
        with self.assertRaises(Exception) as e:
            cfg.a
        self.assertEqual(str(e.exception), "Environment variable A_TEST_ENV_VAR2 is required")


if __name__ == "__main__":
    unittest.main()
