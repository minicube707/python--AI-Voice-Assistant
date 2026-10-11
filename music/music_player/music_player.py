"""
music/music_player/music_player.py
------------------------
Local music player based on python-vlc.
Receives a pre-structured Command.
"""

from __future__ import annotations
import logging
from pathlib import Path

import os
import vlc

from nlp.command import Command

logger = logging.getLogger("music_player")


class MusicPlayer:
    
    def __init__(self, music_dir: str = "."):
        self.music_dir = Path(music_dir)
        self._audio_player = None
        self._is_playing = False
        self.last_command = None
        self._media_list_player = None

    def play(self, command: Command) -> dict:
       
        list_song = command.argument['list_song']

        media_list = vlc.MediaList()

        for song in list_song:
            file_path = self.music_dir / song["path"]

            if not os.path.isfile(file_path):
                logger.error("Fichier non trouvé : %s", file_path)
                continue

            media_list.add_media(str(file_path))

        if media_list.count() == 0:
            return {
                "status": "error",
                "message": "Aucun fichier audio valide"
            }

        self.stop()

        self._media_list_player = vlc.MediaListPlayer()
        self._media_list_player.set_media_list(media_list)
        self._audio_player = self._media_list_player.get_media_player()

        self._media_list_player.play()
        self._is_playing = True
        self.last_command = list_song.copy()

        return {
            "status": "playing",
            "message": "Lecture de la playlist démarrée"
        }


    def stop(self):
        """Stops the currently playing track.

        If a VLC player is initialized, playback is stopped. Otherwise, the
        function does nothing.
        """
        if self._media_list_player is not None:
            self._media_list_player.stop()
        elif self._audio_player is not None:
            self._audio_player.stop()

        self._is_playing = False


    def pause(self):
        """Pauses the currently playing track."""
        if self._audio_player is not None:
            self._is_playing = False
            self._audio_player.pause()


    def resume(self):
        """Resumes the currently paused track."""
        if self._audio_player is not None:
            self._is_playing = True
            self._audio_player.play()


    def set_volume(self, pourcent: int):
        """Set the sound volume"""
        if self._audio_player is not None:
            self._audio_player.audio_set_volume(pourcent)