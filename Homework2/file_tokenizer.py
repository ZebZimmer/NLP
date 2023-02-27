import nltk
import os
import string
import re, pprint, string
from nltk import word_tokenize, sent_tokenize

string.punctuation = string.punctuation + "“" + "”" + "-" + "’" + "‘" + "—"
string.punctuation = string.punctuation.replace(".", "")
from nltk.corpus import stopwords
from typing import Dict, List, Tuple

AUSTEN = 0
DICKENS = 1
TOLSTOY = 2
WILDE = 3

# nltk.download('punkt')
# tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Load stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words('english'))


def process_sentence(sentence: str) -> List[str]:
    """Process the string to remove punctuation, make it lower case,
    tokenize it by word, and remove all stopwords."""
    # code from ChatGPT below to process sentences.
    # Remove punctuation
    sentence = sentence.translate(str.maketrans("", "", string.punctuation))

    # Lowercase sentence
    sentence = sentence.lower()

    # Tokenize sentence into words
    words = nltk.word_tokenize(sentence)

    # Remove stopwords
    # words = [word for word in words if word not in stop_words]
    return words


def parse_file(filename: str, encoding: str) -> List[List[str]]:
    """Parse a txt file and return a list of strings.
    Encoding should be either "utf-8" or "ascii" depending on file type."""
    # ChatGPT code. Make sure encoding is a valid type.
    if encoding.lower() not in ["ascii", "utf8", "utf-8"]:
        raise ValueError(
            f"Unsupported encoding: {encoding}. Only 'ascii' and 'utf8' are supported."
        )

    # append "_utf8" to the filename when necessary.
    if encoding == "utf-8" or encoding == "utf8":
        basename, extension = os.path.splitext(filename)
        filename = basename + "_utf8" + extension

    with open(filename, encoding=encoding) as f:
        text = f.read()

    sentences = nltk.sent_tokenize(text)
    processed_sentences = [process_sentence(sentence) for sentence in sentences]
    return processed_sentences


def tokenize_files(authorlistFilename: str) -> Dict[str, List[List[str]]]:
    """Parse the local authorlist.txt file into lists of sentences.
    Returns a dictionary as follows. However, only authors specified in authorListFilename
    will be present in the dictionary.

    {
        'austen': austenLines,
        'dickens': dickensLines,
        'tolstoy': tolstoyLines,
        'wilde': wildeLines,
    }"""
    print("\nTokenizing and parsing files author files...")
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
    encoding = 'utf8'

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

    print(
        f"lens: wildeLines - {len(wildeLines)}, austen: {len(austenLines)}, tolstoy: {len(tolstoyLines)}, dickens: {len(dickensLines)}"
    )
    
    print(f'austenLines[0]: {austenLines[0]}')
    print(f"dickensLines[0]: {dickensLines[0]}")
    print(f"tolstoyLines[0]: {tolstoyLines[0]}")
    print(f"wildeLines[0]: {wildeLines[0]}")

    # ChatGPT code below to generate authors_dict (our return value).
    authors_dict = {
        "austen": austenLines,
        "dickens": dickensLines,
        "tolstoy": tolstoyLines,
        "wilde": wildeLines,
    }

    authors_dict = {k: v for k, v in authors_dict.items() if v}
    return authors_dict
