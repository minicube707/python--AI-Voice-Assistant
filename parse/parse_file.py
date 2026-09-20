import json
import os
from pathlib import Path

from nlp.utils import normalize


def parse_file(music_path: str):
    
    music_dir = Path(music_path)
    musics = []

    extensions = {".mp3", ".m4a", ".wav", ".flac", ".ogg"}

    for artist_dir in music_dir.iterdir():
        if not artist_dir.is_dir():
            continue

        artist = artist_dir.name

        for music_file in artist_dir.iterdir():
            if not music_file.is_file():
                continue

            if music_file.suffix.lower() not in extensions:
                continue
            
            musics.append({
                "title": normalize(music_file.stem),
                "artist": normalize(artist),
                "file": str(music_file.relative_to(music_dir))
            })

    print("Sucess parsing")
    
    output_path = "data/music_catalog.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(musics, file, ensure_ascii=False, indent=2)

    print(f"Parsing create at {output_path}")