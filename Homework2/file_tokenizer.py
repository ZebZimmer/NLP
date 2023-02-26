import nltk
import os
from typing import Dict, List

AUSTEN = 0
DICKENS = 1
TOLSTOY = 2
WILDE = 3

# nltk.download('punkt')
# tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def parse_file(filename: str, encoding: str) -> List[List[str]]:
    """Parse a txt file and return a list of strings.
    Encoding should be either "utf-8" or "ascii" depending on file type."""
    # append "_utf8" to the filename when necessary.
    if encoding == "utf-8":
        basename, extension = os.path.splitext(filename)
        filename = basename + "_utf8" + extension

    with open(filename, "r", encoding=encoding) as f:
        text = f.read()

    # get a list of sentences from the text.
    sentences = nltk.sent_tokenize(text)

    # Each item in tokenized_sentences is a list of strings itself.
    # For example, tokenized_sentences[0] for austen is: ['family', 'may', 'be', '.']
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    
    return tokenized_sentences


def tokenize_files(authorlistFilename: str) -> Dict[str, List[str]]:
    """Parse the local authorlist.txt file into lists of sentences.
    Returns a dictionary as follows. However, only authors specified in authorListFilename
    will be present in the dictionary.

    {
        'Austen': austenLines,
        'Dickens': dickensLines,
        'Tolstoy': tolstoyLines,
        'Wilde': wildeLines,
    }"""

    encoding = "ascii"
    authorList = [False, False, False, False]
    with open(authorlistFilename, "r") as f:
        for line in f:
            if "#" in line:  # ignore commented out lines in the author list file
                continue
            if "utf8" in line:  # switch to utf8 encoding
                encoding = "utf-8"
            if "austen" in line:
                authorList[AUSTEN] = True
            if "dickens" in line:
                authorList[DICKENS] = True
            if "tolstoy" in line:
                authorList[TOLSTOY] = True
            if "wilde" in line:
                authorList[WILDE] = True

    austenLines = []
    dickensLines = []
    tolstoyLines = []
    wildeLines = []

    if authorList[AUSTEN]:
        # We need the Austen Lines
        austenLines = parse_file("ngram_authorship_train/austen.txt", encoding)

    if authorList[DICKENS]:
        # We need the dickens Lines
        dickensLines = parse_file("ngram_authorship_train/dickens.txt", encoding)

    if authorList[TOLSTOY]:
        # We need the tolstoy Lines
        tolstoyLines = parse_file("ngram_authorship_train/tolstoy.txt", encoding)

    if authorList[WILDE]:
        # We need the wilde Lines
        wildeLines = parse_file("ngram_authorship_train/wilde.txt", encoding)

    print(f"Used {encoding} to read and tokenize files")
    print(
        f"lens: wildeLines - {len(wildeLines)}, austen: {len(austenLines)}, tolstoy: {len(tolstoyLines)}, dickens: {len(dickensLines)}"
    )
    print(f"wildeLines[0]: {wildeLines[0]}")

    # ChatGPT code below to generate authors_dict (our return value).
    authors_dict = {
        "Austen": austenLines,
        "Dickens": dickensLines,
        "Tolstoy": tolstoyLines,
        "Wilde": wildeLines,
    }

    authors_dict = {k: v for k, v in authors_dict.items() if v}
    return authors_dict
