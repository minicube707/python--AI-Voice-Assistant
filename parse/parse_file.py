import json
import os
from pathlib import Path
from mutagen import File

from nlp.utils import normalize

# Supported audio file extensions (compared in lowercase)
EXTENSIONS = {".mp3", ".m4a", ".wav", ".flac", ".ogg"}
OUTPUT_PATH = Path("data/music_catalog.json")


def latest_mtime(artist_dir: Path) -> float:
    """Return the most recent mtime of an artist folder and its album subfolders.

    Adding a file inside an album folder only updates that album's mtime,
    not the parent's, so both levels must be checked.
    """
    latest = artist_dir.stat().st_mtime
    with os.scandir(artist_dir) as it:
        for entry in it:
            if entry.is_dir():
                latest = max(latest, entry.stat().st_mtime)

    return latest


def scan_audio_files(
        folder: Path, music_dir: Path, artist: str, album: str | None
    ) -> list[dict]:
    """Return catalog entries for the audio files located directly in `folder`."""

    entries = []

    with os.scandir(folder) as it:
        for entry in it:

            # Skip subfolders and anything that is not a regular file
            if not entry.is_file():
                continue

            path = Path(entry.path)

            # Skip non-audio files
            if path.suffix.lower() not in EXTENSIONS:
                continue
            
            # Reading metadata
            audio = File(path, easy=True)

            track_number = 0

            if audio and audio.tags:
                track_tag = audio.tags.get("tracknumber")

                if track_tag:
                    try:
                        track_number = int(str(track_tag[0]).split("/")[0])
                    except (ValueError, TypeError):
                        track_number = 0

            entries.append({
                "title": normalize(path.stem),
                "artist": artist,
                "album": album,
                "track_number": track_number,
                "path": str(path.relative_to(music_dir)),
            })

    # Sort by track number
    entries.sort(key=lambda song: song["track_number"])

    return entries


def scan_artist(artist_dir: Path, music_dir: Path) -> list[dict]:
    """Scan one artist folder: loose tracks first, then each album subfolder."""

    # Normalize the artist name once instead of once per track
    artist = normalize(artist_dir.name)

    # Tracks placed directly in the artist folder (no album)
    entries = scan_audio_files(artist_dir, music_dir, artist, album=None)

    # One level of subfolders = albums (sorted for a deterministic order)
    with os.scandir(artist_dir) as it:
        album_dirs = sorted(
            (Path(e.path) for e in it if e.is_dir()), key=lambda p: p.name
        )

    for album_dir in album_dirs:
        album = normalize(album_dir.name)
        entries.extend(scan_audio_files(album_dir, music_dir, artist, album))

    return entries


def load_catalog() -> list[dict]:
    """Load the existing catalog, or return an empty list if missing/corrupted."""
    try:
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def parse_file(music_path: str, force: bool = False):
    """Build or incrementally update the music catalog.

    Only artist folders (or their album subfolders) modified since the last
    catalog write are rescanned. Use force=True to ignore the cache and rescan
    everything (e.g. after changing the behavior of `normalize`).
    """
    
    music_dir = Path(music_path)

    # Ignore the old catalog when forcing a full rescan
    old_catalog = [] if force else load_catalog()

    # Catalog written by an older version (no "album" field): rebuild it fully
    if any("album" not in m for m in old_catalog):
        old_catalog = []

    # A modification time of 0 means every folder will be considered "newer"
    catalog_mtime = OUTPUT_PATH.stat().st_mtime if old_catalog else 0.0

    # Group cached entries by artist folder (first part of the relative path)
    cached: dict[str, list[dict]] = {}
    for m in old_catalog:
        cached.setdefault(Path(m["path"]).parts[0], []).append(m)

    # Sorted for a deterministic catalog order
    artist_dirs = sorted(
        (d for d in music_dir.iterdir() if d.is_dir()), key=lambda d: d.name
    )

    # Artists present in the cache but no longer on disk
    removed = set(cached) - {d.name for d in artist_dirs}

    # Rescan an artist if it is new or if it or any album folder changed
    # since the last run
    to_scan = {
        d.name for d in artist_dirs
        if d.name not in cached or latest_mtime(d) > catalog_mtime
    }

    # Early exit: nothing changed, no need to rewrite the file
    if old_catalog and not to_scan and not removed:
        print("Catalog is up to date, nothing to do")
        return

    # Merge fresh scans with cached entries
    musics = []
    for d in artist_dirs:
        if d.name in to_scan:
            musics.extend(scan_artist(d, music_dir))
        else:
            musics.extend(cached[d.name])


    # Atomic write: dump to a temp file, then replace the target in one step,
    # so a crash mid-write can never leave a corrupted JSON behind
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = OUTPUT_PATH.with_suffix(".json.tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(musics, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, OUTPUT_PATH)

    print(f"{len(to_scan)} artist(s) rescanned, {len(removed)} removed, "
          f"{len(musics)} tracks -> {OUTPUT_PATH}")