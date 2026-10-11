
def filter_by_category(
        list_song: list[dict],
        category: str,
        value: str,
    ) -> dict | None:


    if not list_song or value is None:
        return list_song

    filtered = [
        song
        for song in list_song
        if song[category] == value
    ]

    return filtered


def find_track(
        catalog: list[dict], 
        title: str,
        album: str, 
        artist: str
    ) -> None | list[dict]:
    """Finds a track in the catalog based on the provided metadata.

    If neither a title nor an artist is provided, no track is returned.
    When an artist is provided, the catalog is filtered by artist.
    If a title is also provided, the best matching track is searched among
    the filtered candidates.

    Args:
        catalog (list[dict]): List of dictionnary containing local music.
        title (str): The title of the track to find, or `None` if not provided.
        album (str): The album of the track to find, or `None` if not provided.
        artist (str): The artist of the track to find, or `None` if not provided.

    Returns:
        None | list[dict]: The matching track, all sound that corresponding 
        to the request
    """

    candidates = filter_by_category(catalog, 'artist', artist)
    candidates = filter_by_category(candidates, 'album', album)
    candidates = filter_by_category(candidates, 'title', title)

    if title and candidates:
        #if the title is requested, return only the first song
        return candidates[:1]

    if candidates:
        return candidates
    return None