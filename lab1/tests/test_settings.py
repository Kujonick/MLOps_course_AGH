from src.settings import Settings


def test_settings():
    settings = Settings()
    assert settings.APP_NAME == "MLOPStest"
    assert settings.ENVIRONMENT == "test"
