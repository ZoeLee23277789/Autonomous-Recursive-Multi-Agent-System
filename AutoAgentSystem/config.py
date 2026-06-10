import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==== core ====
AUTO_AGENT_HOME = Path(os.getenv("AUTO_AGENT_HOME", PROJECT_ROOT / ".auto_agent")).expanduser()

# cache directory for optional runtime data
AUTO_AGENT_CACHE_DIR = Path(os.getenv("AUTO_AGENT_CACHE", PROJECT_ROOT / ".cache" / "auto_agent")).expanduser()

# log instances to .auto_agent/instances by default
DEFAULT_LOG_DIR = AUTO_AGENT_HOME / "instances"
