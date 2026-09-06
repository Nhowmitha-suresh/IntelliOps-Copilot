from app.config import settings

def test_settings_defaults():
    assert settings.max_retries == 3
    assert settings.request_timeout_seconds == 30
    assert settings.llm_primary_provider in ["openai", "google"]
