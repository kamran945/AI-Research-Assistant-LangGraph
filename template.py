import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

PROJECT_REPO = "Research-Assistant"

file_list = [
    ".github/workflows/.gitkeep",
    "src/__init__.py",
    "src/agents/__init__.py",
    "src/agents/utils.py",
    "src/backend/__init__.py",
    "src/frontend/__init__.py",
    ".env",
    "app.py",
    "setup.py",
    "research/trials.ipynb",
    "pyproject.toml",
]

for filepath in file_list:
    filepath = Path(filepath)
    directory, filenames = os.path.split(filepath)

    if directory != "":
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Created directory {directory}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as file:
            pass
            logging.info(f"Created file {filepath}")

    else:
        logging.info(f"File {filepath} already exists")