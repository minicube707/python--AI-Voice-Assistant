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


    def play(self, command: Command) -> dict:
        """Finds and plays a track from the music catalog.

        The track is searched using the provided title and artist. If a matching
        track is found, its audio file is loaded and played. The function returns
        a status dictionary indicating whether the track was successfully played,
        could not be found, or could not be accessed.

        Args:
            path (str): path to the music file

        Returns:
            dict: A dictionary containing the playback status and track metadata.
                The status can be `"playing"`, or `"error"`.
        """

        file_path = self.music_dir / command.argument['path']
        title = command.argument['title']
        artist = command.argument['artist']

        if not os.path.isfile(file_path):
            logger.error("Erreur fichier non trouvé")
            return {"status": "error", "message": "Erreur fichier non trouvé"}
            
        logger.info("Lecture : %s - %s (%s)", title, artist, file_path)

        self.stop()
        self._audio_player = vlc.MediaPlayer(str(file_path))
        self._audio_player.play()
        self._is_playing = True
        self.last_command = command

        return {"status": "playing", "title": title, "artist": artist}


    def stop(self):
        """Stops the currently playing track.

        If a VLC player is initialized, playback is stopped. Otherwise, the
        function does nothing.
        """
        if self._audio_player is not None:
            self._is_playing = False
            self._audio_player.stop()


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