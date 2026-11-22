"""
nlp/intent.py
--------------
Intent detection using a list of keywords/trigger verbs.
Deliberately simple: a dictionary `{intent: [triggers]}` is more than
sufficient for an initial set of commands and has negligible CPU/RAM overhead.
"""


INTENT_DESCRIPTIONS = {
    "music_control": (
        "pause the music, "
        "resume the music, "
        "stop the music, "
        "increase the music volume, "
        "decrease the music volume"
    ),
    "play_music": (
        "play a song, artist, or playlist"
    ),
    "exit": (
        "end the conversation"
    ),
    "reply": (
        "reply to a message or use a predefined conversational response"
    ),
    "ai_agent": (
       "send a request to Ollama"
    ),
}