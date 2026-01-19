from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppConfig(BaseSettings):
    artifacts_dir: Path = Field(default=Path("artifacts"), description="Path to artifacts")
    feature_set_name: str = Field(default="test_features", description="Feature set name")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


app_config = AppConfig()