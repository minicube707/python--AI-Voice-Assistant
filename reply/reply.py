
from nlp.command import Command
from speech.tts import TextToSpeech
from music.music_player.music_player import MusicPlayer
import re

REPLY = [
    {
        'quote': 'now say my name',
        'reply': 'your are Heisenberg',
        'language': 'en'
    },
    {
        'quote': 'execute order 66',
        'reply': 'It will be done, My Lord.',
        'language': 'en'
    },
    {
        'quote': 'why mr pink',
        'reply': "because you're a bitch !",
        'language': 'en'
    },
    {
        'quote': "c est une bonne situation ca scribe",
        'reply': "Mais, vous savez, moi je ne crois pas qu il y ait de bonne ou de mauvaise situation. "
        "Moi, si je devais résumer ma vie aujourd hui avec vous, je dirais que c est d abord des rencontres, "
        "des gens qui m ont tendu la main, peut-être à un moment où je ne pouvais pas, où j étais seul chez moi. "
        "Et c est assez curieux de se dire que les hasards, les rencontres forgent une destinée… "
        "Parce que quand on a le goût de la chose, quand on a le goût de la chose bien faite, le beau geste, "
        "parfois on ne trouve pas l interlocuteur en face, je dirais, le miroir qui vous aide à avancer. "
        "Alors ce n est pas mon cas, comme je le disais là, puisque moi au contraire, j ai pu ; et je dis merci à la vie, "
        "je lui dis merci, je chante la vie, je danse la vie… Je ne suis qu amour ! "
        "Et finalement, quand beaucoup de gens aujourd hui me disent : « Mais comment fais-tu pour avoir cette humanité ? » "
        "Eh bien je leur réponds très simplement, je leur dis que c est ce goût de l amour, "
        "ce goût donc qui m a poussé aujourd hui à entreprendre une construction mécanique, mais demain, "
        "qui sait, peut-être simplement à me mettre au service de la communauté, à faire le don, le don de soi…",
        'language': 'fr'
    },
    {
        'quote': 'presente toi',
        'reply': "Bonjour, je suis Jarvis, votre assistant vocal personnel. "
        "Je peux écouter vos demandes, répondre à vos questions et mettre de la musique pour vous accompagner. ",
        'language': 'fr'
    },
    {
        'quote': 'introduce yourself',
        'reply': "Hello, I m Jarvis, your personal voice assistant. "
        "I can listen to your requests, answer your questions, and play music for you. ",
        'language': 'en'
    },
    {
        'quote': 'aie',
        'reply': "Hello. I am Baymax, your personal healthcare companion. "
        "I was alerted to the need for medical attention when you said, 'Aie.'",
        'language': 'en'
    },
]

def extract_reply(text: str) -> Command:
    
    for x in REPLY:

        if re.search(rf"\b{x['quote']}\b", text):

            return Command(
                argument=x,
                status='ok'
            )


def reply_to_the_quote(command: Command, player: MusicPlayer, tts: TextToSpeech) -> dict:

    resume = False
    if player._is_playing:
        resume = True

    player.pause()
    if command.argument['language'] == 'en':
        tts.speak_english(command.argument['reply'])

    elif command.argument['language'] == 'fr':
        tts.speak_french(command.argument['reply'])

    if resume:
        player.resume()

    return {
            "status": "ok",
            "reply": f"reply to the quote: {command.argument['quote']}"
        }