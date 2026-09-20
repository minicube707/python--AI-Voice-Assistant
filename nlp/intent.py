"""
nlp/intent.py
--------------
Intent detection using a list of keywords/trigger verbs.
Deliberately simple: a dictionary `{intent: [triggers]}` is more than
sufficient for an initial set of commands and has negligible CPU/RAM overhead.
"""


from reply.reply import REPLY
from nlp.utils import normalize

# Verbs / phrases that trigger the "play_music" intent
PLAY_MUSIC_TRIGGERS = [
    r"\bmets?(?:-moi)?\b",
    r"\bjoues?(?:-moi)?\b",
    r"\blance\b",
    r"\bdemarre\b",
    r"\bje veux ecouter\b",
    r"\becoute(?:r)?\b",
    r"\bpasse(?:-moi)?\b",
    r"\bencore\b",
    r"\bremets\b",
    r"\bpeux tu mettre\b"
]

# Verbs / phrases that trigger the "music_controls " intent
MUSIC_CONTROLE_TRIGGERS = [
    r"\bstop\b",
    r"\bpause\b",
    r"\breprend\b",
    r"\bcontinue\b",
    r"\bson\b",
    r"\bvolume\b",
    r"\bset\b",   
]

EXIT_TRIGGERS = [
    r"\bau revoir\b",
    r"\bbye bye\b",
    r"\bciao\b", 
]

AI_AGENT_TRIGGERS = [
    r"\bdemande a olama\b",
]

# Future intentions (roadmap): pause_music, next_track, volume_up, ...
# Simply add an entry here + a dedicated extractor if needed.
INTENTS = {
    'music_control': MUSIC_CONTROLE_TRIGGERS,
    'play_music': PLAY_MUSIC_TRIGGERS,
    'exit': EXIT_TRIGGERS,
    'reply': [rf"\b{x['quote']}\b" for x in REPLY],
    'ai_agent': AI_AGENT_TRIGGERS
}


INTENT_DESCRIPTIONS = {
    "music_control": (
        "contrôler la musique actuellement en lecture, "
        "par exemple mettre en pause, reprendre, arrêter ou modifier le volume"
    ),
    "play_music": (
        "demander de jouer, lancer ou écouter une musique, "
        "une chanson, un artiste ou une playlist"
    ),
    "exit": (
        "quitter l'assistant, terminer la conversation ou dire au revoir"
    ),
    "reply": (
        "répondre à un message ou utiliser une réponse conversationnelle prédéfinie"
    ),
    "ai_agent": (
        "demander à un autre agent d'intelligence artificielle de répondre "
        "à une question ou d'effectuer une tâche. La pharse commence toujours par:"
        "'demande a olama'"
    ),
}