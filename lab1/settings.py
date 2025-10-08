from pydantic_settings import BaseSettings
from pydantic import field_validator


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
