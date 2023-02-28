import nltk, os, string, random
from typing import Dict, List, Tuple, Union
from nltk.corpus import stopwords

string.punctuation = string.punctuation + "“" + "”" + "-" + "’" + "‘" + "—"

AUSTEN = 0
DICKENS = 1
TOLSTOY = 2
WILDE = 3

# Load stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


def generate_testfile(
    authorlist_filename: str,
    testfile_name: str,
    num_lines: int = 100,
) -> bool:
    """Create a testfile, which consists of randomly selected lines from all authors.
    These lines have been preprocessed using tokenize_files().
    Return true when file has been created, False otherwise.
    Each line in the file followws this format:\n
    f"{line}@{author}"
    """
    if os.path.exists(testfile_name):
        print(f"\n{testfile_name} already exists. Did not create new file.")
        return False

    # get all the untokenized sentences into an authors dict.
    author_list, encoding = get_authors_and_encoding(authorlist_filename)
    print(f"\nCreating {testfile_name} with {encoding} encoding")
    if author_list[AUSTEN]:
        austen_untokenized = parse_file("ngram_authorship_train/austen.txt", encoding, untokenized=True)

    if author_list[DICKENS]:
        dickens_untokenized = parse_file("ngram_authorship_train/dickens.txt", encoding, untokenized=True)

    if author_list[TOLSTOY]:
        tolstoy_untokenized = parse_file("ngram_authorship_train/tolstoy.txt", encoding, untokenized=True)

    if author_list[WILDE]:
        wilde_untokenized = parse_file("ngram_authorship_train/wilde.txt", encoding, untokenized=True)

    authors_dict = {
        "austen": austen_untokenized,
        "dickens": dickens_untokenized,
        "tolstoy": tolstoy_untokenized,
        "wilde": wilde_untokenized,
    }

    authors_dict = {k: v for k, v in authors_dict.items() if v}

    # code from ChatGPT
    with open(testfile_name, "w", encoding=encoding) as f:
        for i in range(num_lines):
            author = random.choice(list(authors_dict.keys()))
            line = random.choice(authors_dict[author])
            line = line.replace('\n', '')
            if i < num_lines - 1:
                f.write(f"{line}@{author}\n")
            else:
                f.write(f"{line}@{author}")
    return True


def process_sentence(sentence: str, stop: bool) -> List[str]:
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
    if stop:
        words = [word for word in words if word not in stop_words]

    return words


def parse_file(
    filename: str, encoding: str, untokenized: bool = False
) -> Union[List[List[str]], List[str]]:
    """Parse a txt file and return a list of tokenized sentences.
    Encoding should be either "utf-8" or "ascii" depending on file type.
    If parameter untokenized is true, then this function will return a list of untokenized strings.
    """
    # ChatGPT code. Make sure encoding is a valid type.
    if encoding.lower() not in ["ascii", "utf-8"]:
        raise ValueError(
            f"Unsupported encoding: {encoding}. Only 'ascii' and 'utf-8' are supported."
        )

    # append "_utf8" to the filename when necessary.
    if encoding == "utf-8":
        basename, extension = os.path.splitext(filename)
        filename = basename + "_utf8" + extension

    print(
        f"Parsing {filename} using {encoding}. Returning {'untokenized' if untokenized else 'tokenized'}"
    )
    with open(filename, encoding=encoding) as f:
        text = f.read()

    sentences = nltk.sent_tokenize(text)
    if untokenized:
        return sentences

    preprocessed_sentences = [
        process_sentence(sentence, stop=False) for sentence in sentences
    ]

    return preprocessed_sentences

def get_authors_and_encoding(authorlist_filename: str) -> Tuple[List[str], str]:
    """Return a tuple of: (list of authors available, file encoding)"""
    encoding = "ascii"
    author_list = [False, False, False, False]
    with open(authorlist_filename, "r") as f:
        for line in f:
            if "#" in line:  # ignore commented out lines in the author list file
                continue
            if "utf8" in line:  # switch to utf8 encoding
                encoding = "utf-8"
            if "austen" in line:
                author_list[AUSTEN] = True
            if "dickens" in line:
                author_list[DICKENS] = True
            if "tolstoy" in line:
                author_list[TOLSTOY] = True
            if "wilde" in line:
                author_list[WILDE] = True
    return author_list, encoding

def tokenize_files(
    authorlist_filename: str,
) -> Dict[str, List[List[str]]]:
    """Parse the local authorlist.txt file into lists of sentences.
    Will generate a testfile with testfile_num_lines if testfile_name is specified.
    Returns a dictionary as follows. However, only authors specified in authorlist_filename
    will be present in the dictionary.

    {
        'austen': austenLines,
        'dickens': dickensLines,
        'tolstoy': tolstoyLines,
        'wilde': wildeLines,
    }"""
    print("\nTokenizing and parsing files author files...")


    austenLines = []
    dickensLines = []
    tolstoyLines = []
    wildeLines = []

    author_list, encoding = get_authors_and_encoding(authorlist_filename)
    if author_list[AUSTEN]:
        # We need the Austen Lines
        austenLines = parse_file("ngram_authorship_train/austen.txt", encoding)

    if author_list[DICKENS]:
        # We need the dickens Lines
        dickensLines = parse_file("ngram_authorship_train/dickens.txt", encoding)

    if author_list[TOLSTOY]:
        # We need the tolstoy Lines
        tolstoyLines = parse_file("ngram_authorship_train/tolstoy.txt", encoding)

    if author_list[WILDE]:
        # We need the wilde Lines
        wildeLines = parse_file("ngram_authorship_train/wilde.txt", encoding)

    # ChatGPT code below to generate authors_dict (our return value).
    authors_dict = {
        "austen": austenLines,
        "dickens": dickensLines,
        "tolstoy": tolstoyLines,
        "wilde": wildeLines,
    }

    authors_dict = {k: v for k, v in authors_dict.items() if v}
    return authors_dict
