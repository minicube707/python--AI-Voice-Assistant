
import json
import re
from pathlib import Path

from nlp.command import Command
from speech.tts import TextToSpeech
from music.music_player.music_player import MusicPlayer


# Chemin du fichier JSON
REPLY_FILE = Path(__file__).parent / "reply.json"


def REPLY() -> list:
    """Charge les réponses depuis le fichier JSON."""
    with REPLY_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def extract_reply(text: str) -> Command:

    for item in REPLY():

        if re.search(
            rf"\b{re.escape(item['quote'])}\b",
            text,
            re.IGNORECASE
        ):
            return Command(
                argument=item,
                status="ok"
            )


def reply_to_the_quote(
    command: Command,
    player: MusicPlayer,
    tts: TextToSpeech
) -> dict:

    resume = player._is_playing

    player.pause()

    if command.argument["language"] == "en":
        tts.speak_english(command.argument["reply"])

    elif command.argument["language"] == "fr":
        tts.speak_french(command.argument["reply"])

    if resume:
        player.resume()

    return {
        "status": "ok",
        "reply": f"reply to the quote: {command.argument['quote']}"
    }
