from nlp.command import Command
from music.music_player.music_player import MusicPlayer

def music_player_controls(command: Command, player: MusicPlayer) -> dict:
    """Apply a playback control to the music player.

    Args:
        command (Command): Command containing the playback action to execute.
        player (MusicPlayer): Music player on which to apply the action.

    Returns:
        dict: A result dictionary containing the status of the operation.
            If the action is unknown, the dictionary contains an error
            status and a description.
    """

    action = command.argument.get("action")

    actions = {
        "stop": (player.stop, "stop"),
        "pause": (player.pause, "pause"),
        "continue": (player.resume, "resume"),
    }

    if action not in actions:
        return {
            "status": "error",
            "description": f"unknown action: {action}",
        }

    method, status = actions[action]
    method()

    return {
        "status": status,
    }


def music_volume_controls(command: Command, player: MusicPlayer):
    """Apply a volume control to the music player.

    Args:
        command (Command): Command containing the volume action to execute.
        player (MusicPlayer): Music player on which to apply the action.

    Returns:
        dict: A result dictionary containing the status of the operation
            and, when applicable, the new volume level.
            If no music player is running, the status is ``"nothing"``.
            If the action is invalid or the volume is missing, the status
            is ``"error"``.
    """

    action = command.argument.get("action")

    music_player = player._audio_player

    if music_player is None:
        return {
            "status": "nothing",
            "description": "no music player running",
        }

    current_volume = music_player.audio_get_volume()

    if action == "decrease_volume":
        new_volume = max(0, current_volume - 10)

    elif action == "increase_volume":
        new_volume = min(100, current_volume + 10)

    elif action == "volume":
        volume = command.argument.get("volume")

        if volume is None:
            return {
                "status": "error",
                "description": "volume is required",
            }

        new_volume = max(0, min(100, volume))

    else:
        return {
            "status": "error",
            "description": f"unknown volume action: {action}",
        }

    player.set_volume(new_volume)

    return {
        "status": action,
        "new_volume": new_volume,
    }


def music_controls(command: Command, player: MusicPlayer) -> dict:
    """Apply a music control to the music player.

    This function dispatches the command to the appropriate control handler
    based on the action provided in the command.

    Args:
        command (Command): Command containing the action to execute.
        player (MusicPlayer): Music player on which to apply the action.

    Returns:
        dict: A result dictionary containing the status of the operation.
            The exact contents depend on the selected control handler.
    """
    
    if command.argument == {}:
        return {
                "status": "error",
                "description": "invalid command",
            }

    if command.argument["action"] in ("pause", "stop", "continue"):
        return music_player_controls(command, player)

    if command.argument["action"] in (
        "volume",
        "increase_volume",
        "decrease_volume",
    ):
        return music_volume_controls(command, player)
