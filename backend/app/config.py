from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "灰袍 AIGC 检测"
    app_version: str = "2.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # Chinese-optimized models
    model_primary: str = "Hello-SimpleAI/chatgpt-detector-roberta-chinese"
    model_cache_dir: str = "./models"

    max_text_length: int = 50_000
    chunk_max_tokens: int = 512
    chunk_stride: int = 256

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
