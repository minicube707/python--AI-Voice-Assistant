from nlp.command import Command

import re

def extract_music_controls(sentence: str) -> Command:
    """Extracts the music controls from a sentence.

    The function  attempts to extract the music controls using predefined keywords.

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

        match = re.search(r"(\d+)\s*(?:%)", sentence)

        if match:
            volume = int(match.group(1))
        else:
            volume = None
    
        return Command(
            argument={
                "action": "volume",
                'volume': volume
            }
        )

    return Command(status="error")