"""
music/music_player/extraction.py
--------------
Provides functions for extracting music metadata from text.

This module contains utilities for processing sentences and extracting
information such as music titles and artist names. It handles text
normalization, keyword matching, noise removal, and metadata completion.
"""

from rapidfuzz import fuzz, process

from music.music_player.find_track import find_track, THRESOLD_SCORE
from nlp.command import Command

MUSIC_KEYWORDS = [
    "musique",
    "chanson",
    "titre",
    "morceau",
]

ARTIST_KEYWORDS = [
    "artiste",
    "groupe",
    "chanteur",
    "hanteuse",
]

CONJUNCTION_WORDS = [
    "de",
    "du",
    "par",
    "avec"
]

ARTICLE_WORD = [
    "un",
    "une",
    "la",
    "le",
    "l"
]

COMMAND_KEYWORDS = [
    "mets",
    "met",
    "joue",
    "lance",
    "demarre",
]

NOISE_WORD = [
    "quelque chose",
    "moi",
    "s il te plait",
]


def create_list_artist(catalog: list[dict]) -> list[str]:
    """Creates a list of unique artists from the sorted music catalog.

    The catalog is expected to be sorted by artist. Consecutive duplicate
    artists are ignored while preserving the order in which artists appear.

    Args:
        catalog (list[dict]): List of dictionnary containing local music.

    Returns:
        list[str]: A list of unique artists in catalog order.
    """

    artists = []
    last_artist = None

    for e in catalog:
        artist = e["artist"]
        if artist != last_artist:
            artists.append(artist)
            last_artist = artist

    return artists


def remove_word(sentence: list[str], delete_words: list[str]):
    """Removes specified words from a sentence.

    Args:
        sentence (list[str]): A list of words forming the sentence.
        delete_words (list[str]): A list of words to remove from the sentence.

    Returns:
        None: The `sentence` list is modified in place.
    """

    for word in delete_words:
        if word in sentence:
            sentence.remove(word)



def extract_value(
    sentence: list[str],
    keywords_start: list[str],
    keywords_stop: list[str],
) -> str | None:
    """Extracts a value from a sentence between start and stop keywords.

    Args:
        sentence (list[str]): A list of words forming the sentence.
        keywords_start (list[str]): Keywords that indicate the beginning of the value.
        keywords_stop (list[str]): Keywords that indicate the end of the value.

    Returns:
        str | None: The extracted value as a string, or None if no value is found.

    Notes:
        The extracted words are removed from the `sentence` list.
    """

    res = []
    add = False

    for i, word in enumerate(sentence[:]):
        if word in keywords_start and not add:
            add = True

        if word in keywords_stop and add:
            add = False

        if add:
            res.append(word)
            sentence.remove(word)

    return ' '.join(res[1:]) if res else None


def find_value(sentence: list[str]) -> tuple[str, str]:
    """Splits a sentence into two values based on a conjunction word.

    Args:
        sentence (list[str]): A list of words forming the sentence.

    Returns:
        tuple[str, str]: Two string that could be either the title, either the artist name
    """

    sep_find = False
    val1 = []
    val2 = []

    for word in sentence:

        if word in CONJUNCTION_WORDS:
            sep_find = True
            continue

        elif not sep_find:
            val1.append(word)

        else:
            val2.append(word)

    return ' '.join(val1), ' '.join(val2)


def complete_metadata(
    title: str,
    artist: str,
    sentence: list[str]
    ) -> tuple[str | None, str | None]:
    """Completes missing title or artist metadata using the remaining words.

    Args:
        title (str | None): The title of the item, if available.
        artist (str | None): The artist of the item, if available.
        sentence (list[str]): A list of words used to complete the missing metadata.

    Returns:
        tuple[str | None, str | None]: The completed title and artist.
    """

    words_to_remove = CONJUNCTION_WORDS + ARTICLE_WORD
    remove_word(sentence, words_to_remove)

    if title is None and artist is not None:
        title = ' '.join(sentence)

    elif title is not None and artist is None:
        artist = ' '.join(sentence)

    return title, artist


def get_best_matching(
    titles: list[str],
    artists: list[str],
    value: str,
) -> tuple[tuple[str | None, float], tuple[str | None, float]]:
    """Finds the best matching title and artist for a given value.

    Uses fuzzy string matching to compare the input value against the
    provided titles and artists. The similarity score is the average of
    WRatio and token_sort_ratio.

    Args:
        titles (list[str]): A list of available music titles.
        artists (list[str]): A list of available artist names.
        value (str): The value to compare against the titles and artists.

    Returns:
        tuple[tuple[str | None, float], tuple[str | None, float]]:
            A tuple containing the best title match and the best artist match.
            Each match is represented by a tuple containing the matched string
            and its similarity score. If no match is found, the corresponding
            result is `(None, 0)`.
    """

    def combined_scorer(s1: str, s2: str, **kwargs) -> float:
        wratio = fuzz.WRatio(s1, s2)
        len_s1 = len(s1.split(' '))
        len_s2 = len(s2.split(' '))
        length_ratio = min(len_s1, len_s2) / max(len_s1, len_s2)

        penalty = 0.9 + (length_ratio * 0.1)
        wratio *= penalty
        
        return wratio

    if artists:
        best_artist = process.extractOne(
            value,
            artists,
            scorer=combined_scorer,
        )

    else:
        best_artist = (None, 0)

    if titles:
        best_title = process.extractOne(
            value,
            titles,
            scorer=combined_scorer,
        )

    else:
        best_title = (None, 0)

    if not best_title:
        best_title = (None, 0)

    if not best_artist:
        best_artist = (None, 0)

    return best_title, best_artist


def best_match(
    match1: tuple[str, float],
    match2: tuple[str, float],
) -> str | None:
    """Returns the best match based on similarity scores.

    A match is considered valid only if at least one of the two scores is
    above `THRESOLD_SCORE`. If both scores are equal, no match is selected.

    Args:
        match1 (tuple[str, float]): The first match and its similarity score.
        match2 (tuple[str, float]): The second match and its similarity score.

    Returns:
        str | None: The value of the match with the highest score, or `None`
            if neither match exceeds the score threshold or if both scores
            are equal.
    """

    score1 = match1[1]
    score2 = match2[1]

    if score1 <= THRESOLD_SCORE and score2 <= THRESOLD_SCORE:
        return None

    if score1 == score2:
        return None

    return match1[0] if score1 > score2 else match2[0]


def keep_best_match(
    title: tuple[str, float],
    artist: tuple[str, float],
) -> tuple[tuple[str | None, float], tuple[str | None, float]]:
    """Keeps the match with the highest similarity score.

    Compares the title and artist match scores and keeps only the match with
    the highest score. The other match is replaced with `(None, 0)`.

    Args:
        title (tuple[str, float]): The title match and its similarity score.
        artist (tuple[str, float]): The artist match and its similarity score.

    Returns:
        tuple[tuple[str | None, float], tuple[str | None, float]]:
            A tuple containing the title and artist matches. Only the match
            with the highest score is preserved.
    """
    if title[1] > artist[1]:
        return title, (None, 0)
    return (None, 0), artist


def refine_with_catalog(
    catalog: list[dict],
    title: str, 
    artist: str
    ) -> tuple[str, str]:
    """Refines the data extract from the sentence by matching values against the catalog.

    If the catalog is empty, the original extraction is returned unchanged.
    Otherwise, the extracted values are compared against the catalog titles 
    and artists using fuzzy matching. 
    The best matches are then selected to determine the most likely title and artist.

    Args:
        catalog (list[dict]): List of dictionnary containing local music.
        title (str): The title of the track to find, or `None` if not provided.
        artist (str): The artist of the track to find, or `None` if not provided.

    Returns:
        tuple[str, str]: The refined extraction result containing `title` and
            `artist` when a catalog match is found, or the original extraction
            if no catalog is available or the information was already found.
    """
    
    if not catalog:
        return title, artist

    titles = [e["title"] for e in catalog]
    artists = create_list_artist(catalog)

    best_title1, best_artist1 = get_best_matching(titles, artists, title)
    best_title2, best_artist2 = get_best_matching(titles, artists, artist)

    best_title1, best_artist1 = keep_best_match(best_title1, best_artist1)
    best_title2, best_artist2 = keep_best_match(best_title2, best_artist2)

    title = best_match(best_title1, best_title2)
    artist = best_match(best_artist1, best_artist2)

    return title, artist


def extract_play_music(catalog: list[dict], sentence: str) -> Command:
    """Extracts music information from a sentence.

    The function removes command keywords, and attempts
    to extract the music title and artist using predefined keywords. If either
    the title or artist is missing, it tries to complete the metadata using the
    remaining words.

    Args:
        catalog (list[dict]): List of dictionnary containing local music.
        sentence (str): The input sentence containing music information.

    Returns:
        Command: The command with instruction
    """
    if "encore" == sentence or "remets" in sentence:
        return Command(
        argument={
            'replay': True,
        },
        status="ok"
    )

    #Remove the noisy world, that doesn't help to find the track
    for mot in NOISE_WORD:
        sentence = sentence.replace(mot, '')
    
    sentence = sentence.strip().split(' ')
    remove_word(sentence, COMMAND_KEYWORDS )

    title = extract_value(sentence, MUSIC_KEYWORDS, CONJUNCTION_WORDS)
    artist = extract_value(sentence, ARTIST_KEYWORDS, CONJUNCTION_WORDS)

    if title == None or artist == None:
        title, artist = complete_metadata(title, artist, sentence.copy())
    
    if title == None or artist == None:
        title, artist = find_value(sentence)

    title, artist = refine_with_catalog(catalog, title, artist)

    track = find_track(catalog, title, artist)

    if track is None:
        title=""
        artist=""
        status = "not_found"
        path = ""
    
    else:
        title=track["title"]
        artist=track["artist"]
        status="ok"
        path=track["file"]

    return  Command(
        argument={
            'title': title,
            'artist': artist,
            'path': path,
            'replay': False
        },
        status=status,
    )