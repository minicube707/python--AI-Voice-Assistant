import random

from rapidfuzz import fuzz, process, utils

THRESOLD_SCORE = 80


def filter_by_artist(
    candidates: list[dict],
    artist: str,
) -> list[dict]:
    """Filters track candidates by artist similarity.

    Tracks are kept when their artist name reaches the similarity threshold
    defined by `THRESOLD_SCORE`. If no artist matches the threshold, the
    original list of candidates is returned instead of an empty list.

    Args:
        candidates (list[dict]): The tracks to filter.
        artist (str): The artist name used for filtering.

    Returns:
        list[dict]: The tracks matching the artist, or the original candidates
            if no match is found.
    """

    if not artist:
        return candidates

    filtered = [
        track
        for track in candidates
        if fuzz.token_sort_ratio(track["artist"], artist) >= THRESOLD_SCORE
    ]

    return filtered if filtered else candidates


def find_by_title(
    candidates: list[dict],
    title: str,
) -> dict | None:
    """Finds the track with the closest matching title.

    Uses fuzzy string matching to compare the given title with the titles
    of the candidate tracks. The best match is returned only if its
    similarity score reaches `THRESOLD_SCORE`.

    Args:
        candidates (list[dict]): The tracks to search through.
        title (str): The title to match against the candidate tracks.

    Returns:
        Optional[dict]: The track with the best matching title, or `None`
            if there are no candidates or no match meets the similarity
            threshold.
    """

    if not candidates:
        return None

    best = process.extractOne(
        title,
        [track["title"] for track in candidates],
        scorer=fuzz.token_sort_ratio,
        processor=utils.default_process,
    )

    if not best or best[1] < THRESOLD_SCORE:
        return None

    return next(
        (track for track in candidates if track["title"] == best[0]),
        None,
    )


def pick_random(candidates: list[dict]) -> dict | None:
    """Selects a random track from a list of candidates.

    Args:
        candidates (list[dict]): The tracks to choose from.

    Returns:
        Optional[dict]: A randomly selected track, or `None` if the list
            of candidates is empty.
    """

    if not candidates:
        return None

    return random.choice(candidates)


def find_track(
    catalog: list[dict], 
    title: str, 
    artist: str) -> None | dict:
    """Finds a track in the catalog based on the provided metadata.

    If neither a title nor an artist is provided, no track is returned.
    When an artist is provided, the catalog is filtered by artist.
    If a title is also provided, the best matching track is searched among
    the filtered candidates. Otherwise, a random track is selected from
    the matching artist candidates.

    Args:
        catalog (list[dict]): List of dictionnary containing local music.
        title (str): The title of the track to find, or `None` if not provided.
        artist (str): The artist of the track to find, or `None` if not provided.

    Returns:
        None | dict: The matching track, a random track matching the artist,
            or `None` if no title or artist is provided or no suitable track
            is found.
    """

    # Wrong artist or title given (e.g. "Play unknown artist")
    if title is None and artist is None:
        return None

    candidates = filter_by_artist(catalog, artist)

    if title:
        return find_by_title(candidates, title)

    return pick_random(candidates)