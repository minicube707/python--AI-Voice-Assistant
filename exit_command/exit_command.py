import logging
import sys
import random

from nlp.command import Command
from speech.tts import TextToSpeech 

logger = logging.getLogger("exit_command")


def exit_command(tts: TextToSpeech):

    sentences = [
        {
            'sentence': 'Au revoir, à bientôt !',
            'language': 'fr'
        },
        {
            'sentence': 'À bientôt !',
            'language': 'fr'
        },
        {
            'sentence': 'Passe une bonne journée !',
            'language': 'fr'
        },
        {
            'sentence': 'À la prochaine !',
            'language': 'fr'
        },
        {
            'sentence': 'Bonne journée, à bientôt !',
            'language': 'fr'
        },
        {
            'sentence': 'À plus tard !',
            'language': 'fr'
        },
        {
            'sentence': 'On se retrouve bientôt !',
            'language': 'fr'
        },
        {
            'sentence': "C'était un plaisir, à bientôt !",
            'language': 'fr'
        },
        {
            'sentence': 'À une prochaine fois !',
            'language': 'fr'
        },
        {
            'sentence': 'May the Force be with you',
            'language': 'en'
        },
        {
            'sentence': 'Hasta la vista, baby',
            'language': 'en'
        },
        {
            'sentence': "I'll be back",
            'language': 'en'
        },
    ]


    sentence = random.choice(sentences)

    if sentence['language'] == 'fr':
        tts.speak_french(sentence['sentence'])
    
    elif sentence['language'] == 'en':
        tts.speak_english(sentence['sentence'])

    logger.info("Arrêt demandé par l'utilisateur.")
    sys.exit(0)