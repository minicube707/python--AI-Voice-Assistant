from typing import Callable

from nlp.command import Command

MusicControlsExtractor = Callable[[str], Command]

EXTRACTORS = ("rules", "neural")


def create_music_controls_intent(kind: str = "rules") -> MusicControlsExtractor:
    """Return the function that turns a sentence into a music-control Command.

    Args:
        kind: ``"rules"`` (keywords) or ``"neural"`` (trained network).
    """

    if kind == "rules":
        from music.music_controls.rules import extract_music_controls
        return extract_music_controls

    if kind == "neural":
        # Imported here so torch is only loaded when it is needed
        from music.music_controls.neural import NeuralMusicControls
        return NeuralMusicControls()

    raise ValueError(
        f"Unknown music_controls_intent: {kind!r} (expected one of {EXTRACTORS})"
    )
