"""
music/music_player/extraction.py
--------------
Provides functions for extracting music metadata from text.

This module contains utilities for processing sentences and extracting
information such as music titles and artist names. It handles text
normalization, keyword matching, noise removal, and metadata completion.
"""

from rapidfuzz import fuzz, process

from music.music_player.find_track import find_track
from nlp.command import Command

THRESOLD_SCORE = 75

# Lower bar used when the candidates are already restricted to one artist
# or one album: there are few of them, so a weaker match is still trustworthy.
SCOPED_THRESOLD_SCORE = 60

# Maximum share of the score removed when the two texts have a very
# different number of words ("chuan legacy" vs "tron legacy end titles").
LENGTH_PENALTY = 0.5

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
    "chanteuse",
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


def create_unique_list(catalog: list[dict], key: str) -> list[str]:
    """Creates a list of unique item from the sorted music catalog.

    The catalog is expected to be sorted. Consecutive duplicate
    item are ignored while preserving the order in which item appear.

    Args:
        catalog (list[dict]): List of dictionnary containing local music.

    Returns:
        list[str]: A list of unique item in catalog order.
    """

    unique_list = []
    last_value = None

    for e in catalog:
        value = e[key]
        if value != last_value:
            unique_list.append(value)
            last_value = value

    return unique_list


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


def complete_metadata(
        requested_item: str,
        context_item: str,
        sentence: list[str]
    ) -> tuple[str, str]:
    """Completes missing title or context_item metadata using the remaining words.

    Args:
        requested_item (str): The item who is played.
        context_item (str): The item where is it.
        sentence (list[str]): A list of words used to complete the missing metadata.

    Returns:
        tuple[str, str]: The completed requested_item (title or album) and context_item (album or artist).
    """

    words_to_remove = CONJUNCTION_WORDS + ARTICLE_WORD
    remove_word(sentence, words_to_remove)

    if not requested_item and context_item:
        requested_item = ' '.join(sentence)

    if requested_item and not context_item:
        context_item = ' '.join(sentence)

    return requested_item, context_item


def find_value(sentence: list[str]) -> tuple[str, str]:
    """Splits a sentence without keyword into "requested item" and "context".

    Used for sentences like "joue Billie Jean de Michael Jackson": the text
    is cut at the last conjunction word, because a title may itself contain
    one ("le chant de la terre de ...") while the artist comes last.

    Args:
        sentence (list[str]): A list of words forming the sentence.

    Returns:
        tuple[str, str]: The requested item and the context item
            (the second one is empty when there is no conjunction).
    """

    split = max((i for i, word in enumerate(sentence) if word in CONJUNCTION_WORDS), default=None,)

    if split is None:
        left, right = sentence, []
    else:
        left, right = sentence[:split], sentence[split + 1:]

    def strip_articles(words: list[str]) -> str:
        words = list(words)
        while words and words[0] in ARTICLE_WORD:
            words.pop(0)
        return ' '.join(words)

    return strip_articles(left), strip_articles(right)


def get_best_matching(
        list1: list[str],
        list2: list[str],
        value: str,
    ) -> tuple[
            tuple[str | None, float],
            tuple[str | None, float]
        ]:
    """Finds the best matching for a given value.

    Uses fuzzy string matching to compare the input value against the
    provided list1 and list2. The similarity score is WRatio, reduced by a
    penalty when the number of words or characters differs
    (up to `LENGTH_PENALTY`).
    WRatio alone is too lenient: a single shared word is enough to give a
    very high score to a much longer candidate.

    Args:
        list1 (list[str]): A list of available items.
        list2 (list[str]): A list of available items.
        value (str): The value to compare against the list1 and list2.

    Returns:
        tuple[tuple[str | None, float], tuple[str | None, float]]:
            A tuple containing the best item match and the best item match.
            Each match is represented by a tuple containing the matched string
            and its similarity score. If no match is found, the corresponding
            result is `(None, 0)`.
    """

    def combined_scorer(s1: str, s2: str, **kwargs) -> float:
        wratio = fuzz.WRatio(s1, s2)
        len_s1 = len(s1.split(' '))
        len_s2 = len(s2.split(' '))
        word_ratio = min(len_s1, len_s2) / max(len_s1, len_s2)

        # Also compare the number of characters: with a single word on each
        # side ("la" vs "blablabla"), the word count cannot tell them apart.
        char_ratio = min(len(s1), len(s2)) / max(len(s1), len(s2), 1)
        length_ratio = min(word_ratio, char_ratio)

        penalty = (1 - LENGTH_PENALTY) + (length_ratio * LENGTH_PENALTY)

        return wratio * penalty

    if list1:
        best_list1 = process.extractOne(
            value,
            list1,
            scorer=combined_scorer,
        )
    else:
        best_list1 = (None, 0)

    if list2:
        best_list2 = process.extractOne(
            value,
            list2,
            scorer=combined_scorer,
        )
    else:
        best_list2 = (None, 0)

    if not best_list2:
        best_list2 = (None, 0)

    if not best_list1:
        best_list1 = (None, 0)

    return best_list1, best_list2


def refine_with_catalog(
        catalog: list[dict],
        title: str, 
        album: str,
        artist: str,
        requested_item: str,
        context_item: str
    ) -> tuple[str, str, str]:
    """Refines the data extract from the sentence by matching values against the catalog.

    If the catalog is empty, the original extraction is returned unchanged.
    Otherwise, the extracted values are compared against the catalog titles,
    album and artists using fuzzy matching. 
    The best matches are then selected to determine the most likely title, album and artist.

    Args:
        catalog (list[dict]): List of dictionnary containing local music.
        title (str): The title of the track to find.
        album (str): The album of the track to find.
        artist (str): The artist of the track to find.
        requested_item (str): The item who will be played
        context_item (str): Where the track come from

    Returns:
        tuple[str, str, str]: The refined extraction result containing `title`,
            `album` and `artist` when a catalog match is found, 
            or the original extraction if no catalog is available
    """
    
    if not catalog:
        return title, album, artist

    # What the user explicitly asked for ("la chanson", "l'album", "le groupe"),
    # None when the sentence does not say it.
    if title is not None:
        requested_kind = "title"
    elif album is not None:
        requested_kind = "album"
    else:
        requested_kind = None

    if title is not None and album is not None:
        context_kind = "album"
    elif artist is not None or requested_kind == "album":
        context_kind = "artist"      # an album can only come from an artist
    else:
        context_kind = None

    # 1. Resolve the context first (artist or album): once it is known,
    #    the requested item only has to be searched among its tracks.
    context_type, context_value = None, None
    scope = catalog

    if context_item:
        albums = _unique_values(catalog, "album") if context_kind != "artist" else []
        artists = create_unique_list(catalog, "artist") if context_kind != "album" else []

        best_album, best_artist = get_best_matching(albums, artists, context_item)

        if best_album[1] > THRESOLD_SCORE and best_album[1] > best_artist[1]:
            context_type, context_value = "album", best_album[0]
        elif best_artist[1] > THRESOLD_SCORE and best_artist[1] > best_album[1]:
            context_type, context_value = "artist", best_artist[0]

        if context_type:
            scope = [e for e in catalog if e[context_type] == context_value]

    # 2. Resolve the requested item (title or album), only among the kinds
    #    the user asked for.
    requested_type, requested_value = None, None

    if requested_item:
        threshold = THRESOLD_SCORE if scope is catalog else SCOPED_THRESOLD_SCORE

        titles = [e["title"] for e in scope] if requested_kind != "album" else []
        albums = _unique_values(scope, "album") if requested_kind != "title" else []

        best_title, best_album = get_best_matching(titles, albums, requested_item)

        if best_title[1] > threshold and best_title[1] >= best_album[1]:
            requested_type, requested_value = "title", best_title[0]
        elif best_album[1] > threshold and best_album[1] > best_title[1]:
            requested_type, requested_value = "album", best_album[0]

    # 3. Build the result. Whatever could not be matched keeps its raw value,
    #    so that find_track() fails instead of playing a wrong track.
    if requested_type == "title":
        title, album = requested_value, None
    elif requested_type == "album":
        title, album = None, requested_value
    elif requested_item and title is None and album is None:
        title = requested_item      # nothing matched: not_found, never "everything"

    if context_type == "artist":
        artist = context_value
    elif context_type == "album":
        album, artist = context_value, None

    return title, album, artist


def _unique_values(catalog: list[dict], key: str) -> list[str]:
    """Return the distinct non-empty values of `key`, in catalog order."""

    return list(dict.fromkeys(e[key] for e in catalog if e[key]))


def update_item(
        title: str, 
        album: str,
        artist: str
    ):

    type_playing = ''
    requested_item = ''
    type_source = ''
    context_item = ''
    find_album = False

    if title is not None:
        type_playing = 'title'
        requested_item = title
    elif album is not None:
        type_playing = 'album'
        requested_item = album
        find_album = True

    if album is not None and not find_album:
        type_source = 'album'
        context_item = album
    elif artist is not None:
        type_source = 'artist'
        context_item = artist
    
    return type_playing, requested_item, type_source, context_item


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

    if "encore" == sentence or "remets" in sentence or "rejoue" in sentence:
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
    album = extract_value(sentence, ["album"], CONJUNCTION_WORDS)
    artist = extract_value(sentence, ARTIST_KEYWORDS, CONJUNCTION_WORDS)
    
    _, requested_item, _, context_item = update_item(title, album, artist)

    if not requested_item or not context_item:
        requested_item, context_item = complete_metadata(requested_item, context_item, sentence.copy())

    # No keyword at all ("joue X de Y"): split on the conjunction word
    if not requested_item and not context_item:
        requested_item, context_item = find_value(sentence)

    title, album, artist = refine_with_catalog(
        catalog,
        title,
        album,
        artist,
        requested_item,
        context_item)

    list_song = find_track(catalog, title, album, artist)

    type_playing, requested_item, type_source, context_item = update_item(title, album, artist)

    annonce = {
        'type_playing': type_playing,
        'requested_item': requested_item,
        'type_source': type_source,
        'context_item': context_item
    }

    if not list_song:
        return Command(
            argument={
                'annonce': annonce
            },
            status='not_found',
        )

    return  Command(
        argument={
            'list_song': list_song,
            'replay': False,
            'annonce': annonce
        },
        status='ok',
    )