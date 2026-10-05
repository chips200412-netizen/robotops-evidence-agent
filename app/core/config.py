"""全局配置：从 .env 读取，密钥不硬编码（E4）。

阶段 1 切片 2 起被真实 LLM 调用使用；当前先建立读取通道。
默认接入 DeepSeek（OpenAI 兼容接口），可通过 .env 切换任意 OpenAI 兼容服务。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Settings:
    def __init__(self) -> None:
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
        self.openai_base_url: str = os.getenv(
            "OPENAI_BASE_URL", "https://api.deepseek.com"
        )
        self.openai_model: str = os.getenv("OPENAI_MODEL", "deepseek-chat")
        self.openai_timeout_seconds: str = os.getenv("OPENAI_TIMEOUT_SECONDS", "30")
        self.retrieval_similarity_threshold: float = float(
            os.getenv("RETRIEVAL_SIMILARITY_THRESHOLD", "0.6")
        )
        # 阶段 4 · MySQL（docker-compose 起，宿主机端口 3307）
        self.db_host: str = os.getenv("DB_HOST", "127.0.0.1")
        self.db_port: int = int(os.getenv("DB_PORT", "3307"))
        self.db_user: str = os.getenv("DB_USER", "robot")
        self.db_password: str = os.getenv("DB_PASSWORD", "")
        self.db_name: str = os.getenv("DB_NAME", "robot_support")
        self.embedding_model: str = os.getenv(
            "EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"
        )
        project_root = Path(__file__).resolve().parents[2]
        configured_session_db = Path(
            os.getenv("DIAGNOSTIC_SESSION_DB_PATH", "data/robotops-sessions.sqlite3")
        )
        if not configured_session_db.is_absolute():
            configured_session_db = project_root / configured_session_db
        self.diagnostic_session_db_path: str = str(configured_session_db)


settings = Settings()
