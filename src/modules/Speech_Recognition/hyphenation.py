"""Hyphenation module"""


import logging
import string

from hyphen import Hyphenator, dictools
from tqdm import tqdm

from modules.Speech_Recognition.TranscribedData import TranscribedData
from modules.console_colors import blue_highlighted, red_highlighted

log = logging.getLogger("ultrasinger.hyphenation")

PYPHEN_MAX_LEN = 100   # the hyphenation library raises on longer words (library limit)

def language_check(language="en") -> str | None:
    """Check if language is supported"""

    lang_region = None
    installed = dictools.list_installed()
    installed_region_keys = [i for i in installed if i.startswith(language) and "_" in i]
    try:
        # Try to find installed language with region prediction
        lang_region = next(i for i in installed_region_keys if i == f"{language}_{language.upper()}")
    except StopIteration:
        if installed_region_keys:
            # Take first installed region language
            lang_region = installed_region_keys[0]
        else:
            # Take downloadable language key
            downloadable_key = [i for i in dictools.LANGUAGES if i.startswith(language)]
            downloadable_folder_key = [i for i in downloadable_key if i == language]
            if downloadable_folder_key:
                lang_region = downloadable_key[0]
            else:
                try:
                    # Try to find downloadable language with region prediction
                    lang_region = next(i for i in downloadable_key if i == f"{language}_{language.upper()}")
                except StopIteration:
                    if downloadable_key:
                        # Take first installed region language
                        lang_region = downloadable_key[0]

    if lang_region is None:
        return None

    log.info(f"Hyphenate using language code: {blue_highlighted(lang_region)}"
    )
    return lang_region


def __clean_word(word: str):
    """Remove punctuation from word"""
    cleaned_string = ""
    removed_indices = []
    removed_symbols = []
    for i, char in enumerate(word):
        if char not in string.punctuation and char not in " ":
            cleaned_string += char
        else:
            removed_indices.append(i)
            removed_symbols.append(char)
    return cleaned_string, removed_indices, removed_symbols


def __insert_removed_symbols(separated_array, removed_indices, symbols):
    """Insert symbols into the syllables"""
    result = []
    symbol_index = 0
    i = 0

    # Add removed symbols to the syllables
    for syllable in separated_array:
        tmp = ""
        for char in syllable:
            if i in removed_indices:
                tmp += symbols[symbol_index]
                symbol_index += 1
                i += 1
            tmp += char
            i += 1
        result.append(tmp)

    # Add remaining symbols to the last syllable
    if symbol_index < len(symbols):
        tmp = result[-1]
        for i in range(symbol_index, len(symbols)):
            tmp += symbols[i]
        result[-1] = tmp

    return result


def __create_hyphenator(lang_region: str) -> Hyphenator:
    """Create hyphenator"""
    hyphenator = Hyphenator(lang_region)
    return hyphenator


def hyphenation(word: str, hyphenator: Hyphenator) -> list[str] | None:
    """Hyphenate word"""

    cleaned_string, removed_indices, removed_symbols = __clean_word(word)

    if len(cleaned_string) > PYPHEN_MAX_LEN:
        return None

    syllabus = hyphenator.syllables(cleaned_string)

    length = len(syllabus)
    if length > 1:
        hyphen = []
        for i in range(length):
            hyphen.append(syllabus[i])
        hyphen = __insert_removed_symbols(hyphen, removed_indices, removed_symbols)
    else:
        hyphen = None

    return hyphen


def hyphenate_each_word(
        language: str, transcribed_data: list[TranscribedData]
) -> list[list[str]] | None:
    """Hyphenate each word in the transcribed data."""
    lang_region = language_check(language)
    if lang_region is None:
        log.warning(f"{red_highlighted('Error in hyphenation for language ')} {blue_highlighted(language)}{red_highlighted(', maybe you want to disable it?')}"
        )
        return None

    hyphenated_word = []
    try:
        hyphenator = __create_hyphenator(lang_region)
        for i in tqdm(enumerate(transcribed_data)):
            pos = i[0]
            hyphenated_word.append(
                hyphenation(transcribed_data[pos].word, hyphenator)
            )
    except Exception as e:
        log.warning(f"{red_highlighted('Error in hyphenation for language ')} {blue_highlighted(language)}{red_highlighted(', maybe you want to disable it?')}")
        log.warning(f"  -> {e}")
        return None

    return hyphenated_word
