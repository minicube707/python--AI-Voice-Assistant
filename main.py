"""
main.py
-------
Main file
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from voice_assistant.voice_assistant import VoiceAssistant
from parse.parse_file import parse_file


load_dotenv()


def setup_logging(cfg: dict):
    log_cfg = cfg.get("logging", {})

    log_file = log_cfg.get("file", "assistant.log")

    formatter = logging.Formatter(
        "%(asctime)s "
        "[%(levelname)s] "
        "%(name)s: %(message)s"
    )

    # Terminal: WARNING and higher only
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)

    handlers = [console_handler]

    # File : All logs
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(
            log_file,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        handlers.append(file_handler)

   # The root logger must accept all levels
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=handlers,
    )



def main():

    parser = argparse.ArgumentParser(description="Local voice assistant")

    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--parse", action="store_true", help="Parse the music file")

    args = parser.parse_args()

    # -------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # -------------------------------------------------------------
    # Parse musique
    # -------------------------------------------------------------

    if args.parse:
        parse_file(cfg["player"]["music_dir"])
        return

    # -------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------

    setup_logging(cfg)
    logger = logging.getLogger("main")

    # -------------------------------------------------------------
    # Assistant
    # -------------------------------------------------------------

    logger.info("Creating VoiceAssistant...")
    assistant = VoiceAssistant(cfg)

    # -------------------------------------------------------------
    # Run
    # -------------------------------------------------------------
    assistant.run()


if __name__ == "__main__":
    main()
