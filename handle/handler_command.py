"""
handle/handler_command.py
------------------------
Translates a structured Command (intent=play_music)
into a concrete action on the music player or other
intents (pause_music, next_track, volume_up, ...).
"""

from __future__ import annotations

import logging

from nlp.command import Command

from ai_agent.ai_agent import AIAgent
from exit_command.exit_command import exit_command
from music.controls.music_controls import music_controls
from music.music_player.music_player import MusicPlayer
from reply.reply import reply_to_the_quote
from speech.tts import TextToSpeech

logger = logging.getLogger("handle_command")


def handle_music_controls(command: Command, player: MusicPlayer) -> dict:
    return music_controls(command, player)


def handle_play_music(command: Command, player: MusicPlayer, tts: TextToSpeech) -> dict:

    if command.status == "ok":
        if command.argument['replay']:
            command = player.last_command

        player.pause()

        tts.announce_play_music(
            command.argument['title'],
            command.argument['artist']
        )

        return player.play(command)

    elif command.status == "not_found":
        logger.warning(
            "Aucun morceau trouvé pour title=%s artist=%s",
            command.argument['title'],
            command.argument['artist']
        )

        resume_music = False
        if player._is_playing:
            resume_music = True

        player.pause()
        tts.speak_french("Désolé, je n'ai pas trouve la musique")

        if resume_music:
            player.resume()

        return {
            "status": "ignored",
            "reason": f"music not found: {command.status}"
        }


def handle_exit_command(tts: TextToSpeech):
    exit_command(tts)


def handle_reply(command: Command, player: MusicPlayer, tts: TextToSpeech) -> dict:
    return reply_to_the_quote(command, player, tts)


def handle_error(command: Command,player: MusicPlayer, tts: TextToSpeech) -> dict:

    resume = False
    if player._is_playing:
        resume = True
    
    player.pause()
    tts.speak_french("Désolé, je n'ai pas compris votre requête")
    
    if resume:
        player.resume()
    
    return {"status": "ignored", "reason": f"intent non supporté: {command.intent}"}


def handle_ai_agent(command: Command,
        player: MusicPlayer,
        tts: TextToSpeech,
        ai_agent: AIAgent
    ) -> dict:
    
    answer = ai_agent.call_agent(command.argument['query'])

    resume = False
    if player._is_playing:
        resume = True
    
    player.pause()
    tts.speak_french(answer)

    if resume:
        player.resume()

    return {
        "status": "call_agent",
        "query": command.argument['query'],
        "answer": answer
    }


def handler_command(
        command: Command,
        player: MusicPlayer,
        tts: TextToSpeech,
        ai_agent: AIAgent
    ) -> dict:
    """Apply the action to the differents module.

    Args:
        command (Command): Command containing the action to execute.
        player (MusicPlayer): Music player on which to apply the action.
        tts (TextToSpeech): Model that speech the action of the programm

    Returns:
        dict: A result dictionary containing the status of the operation.
            The exact contents depend on the selected control handler.
    """

    if command.intent == "play_music" :
       return handle_play_music(command, player, tts)

    if command.intent == 'music_control':
        return handle_music_controls(command, player)
    
    if command.intent == 'exit':
        player.pause()
        handle_exit_command(tts)

    if command.intent == 'reply':
        return handle_reply(command, player, tts)

    if command.intent == 'ai_agent':
        return handle_ai_agent(command, player, tts, ai_agent)

    return handle_error(command, player, tts)


