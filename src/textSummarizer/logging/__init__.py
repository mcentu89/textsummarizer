import os
import logging
import sys

LOG_DIR = "logs"
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(lineno)d: %(message)s]"
LOG_FILE_PATH = os.path.join(LOG_DIR, "text_summarizer.log")

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("textSummarizerLogger")