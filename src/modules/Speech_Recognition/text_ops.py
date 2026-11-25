"""Word-level text operations (pure functions on TranscribedData lists)."""

import copy

from modules.Speech_Recognition.TranscribedData import TranscribedData


def add_hyphen_to_data(
        transcribed_data: list[TranscribedData], hyphen_words: list[list[str]]
):
    """Add hyphen to transcribed data return new data list"""
    new_data = []

    for i, data in enumerate(transcribed_data):
        if not hyphen_words[i]:
            new_data.append(data)
        else:
            chunk_duration = data.end - data.start
            chunk_duration = chunk_duration / (len(hyphen_words[i]))

            next_start = data.start
            for j in enumerate(hyphen_words[i]):
                hyphenated_word_index = j[0]
                dup = copy.copy(data)
                dup.start = next_start
                next_start = data.end - chunk_duration * (
                        len(hyphen_words[i]) - 1 - hyphenated_word_index
                )
                dup.end = next_start
                dup.word = hyphen_words[i][hyphenated_word_index]
                dup.is_hyphen = True
                if hyphenated_word_index == len(hyphen_words[i]) - 1:
                    dup.is_word_end = True
                else:
                    dup.is_word_end = False
                new_data.append(dup)

    return new_data


def strip_punctuation(transcribed_data: list[TranscribedData]) -> None:
    """Remove unecessary punctuations from transcribed data"""
    punctuation = ".,"
    for i, data in enumerate(transcribed_data):
        data.word = data.word.translate({ord(i): None for i in punctuation})



remove_unecessary_punctuations = strip_punctuation  # old name
