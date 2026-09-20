from pathlib import Path
from pydub import AudioSegment
from pydub.playback import play

BASE_DIR = Path(__file__).parent
SUB_FLODER = "sound"

def play_activation_sound():
    play(AudioSegment.from_file(BASE_DIR / SUB_FLODER / "activation.mp3", format="mp3"))

def play_deactivation_sound():
    play(AudioSegment.from_file(BASE_DIR / SUB_FLODER / "deactivation.mp3", format="mp3"))