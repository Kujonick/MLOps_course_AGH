from pydantic_settings import BaseSettings
from pydantic import field_validator
import yaml


class Settings(BaseSettings):
    ENVIRONMENT: str
    APP_NAME: str
    super_secret_value: str

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, value):
        # prepare validator that will check whether the value of ENVIRONMENT is in (dev, test, prod)
        if value not in {"dev", "test", "prod"}:
            raise ValueError
        return value

    @staticmethod
    def settings_customise_sources(
        cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        def yaml_config_settings_source():
            """Read values from a YAML config file."""
            try:
                with open("secrets.yaml", "r") as f:
                    data = yaml.safe_load(f)
                return data
            except FileNotFoundError:
                return {}

        return (
            init_settings,
            yaml_config_settings_source,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )
