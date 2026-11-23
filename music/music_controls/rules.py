from nlp.command import Command
from music.music_controls.volume import extract_volume


def extract_music_controls(sentence: str) -> Command:
    """Extracts the music controls from a sentence using keyword rules.

    Args:
        sentence (str): The input sentence containing music information.

    Returns:
        Command: The command with instruction
    """

    if "pause" in sentence:
        return Command(argument={"action": "pause"})

    if "stop" in sentence:
        return Command(argument={"action": "stop"})

    if "reprend" in sentence or "reprendre" in sentence or "continue" in sentence:
        return Command(argument={"action": "continue"})

    if "son" in sentence or "volume" in sentence:

        if "plus fort" in sentence or "augmente" in sentence:
            return Command(argument={"action": "increase_volume"})

        if "moins fort" in sentence or "baisse" in sentence:
            return Command(argument={"action": "decrease_volume"})

        return Command(
            argument={"action": "volume", "volume": extract_volume(sentence)}
        )

    return Command(status="error")
