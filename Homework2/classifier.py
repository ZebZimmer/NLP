import argparse
import random
from typing import Dict, List, Tuple
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm.models import (
    MLE,
    LanguageModel,
    StupidBackoff,
    Lidstone,
)
from file_tokenizer import tokenize_files, parse_file, process_test_file
from tqdm import tqdm

random.seed(1234)


def generate_n_gram_model(
    text: List[List[str]],
    n: int,
    model_type: LanguageModel,
    alpha: float = 0.4,
    gamma: float = 0.5,
) -> LanguageModel:
    """Returns an n-gram model which has been trained on the text provided.
    Text should be a list of split sentences.
    model_type is an nltk language model class which has a supports .fit()."""
    train, vocab = padded_everygram_pipeline(n, text)

    # handle the special models.
    if model_type == StupidBackoff:
        lm = StupidBackoff(alpha=alpha, order=n)

    elif model_type == Lidstone:
        lm = Lidstone(gamma=gamma, order=n)
    
    # otherwise every other model should work here.
    else:
        lm = model_type(n)

    lm.fit(train, vocab)
    return lm


# function from ChatGPT
def generate_list_of_models(
    train_dev_dict: Dict[str, Dict[str, List[List[str]]]],
    n: int,
    model_class: LanguageModel,
    alpha: float = 0.4,
    gamma: float = 0.5,
) -> List[Tuple[str, LanguageModel]]:
    """Return a list of Tuples. Tuple[0] is a string for the author name, like 'austen'.
    Tuple[1] is a trained language model based on each author's respective train data.
    """
    print()
    models = []
    for author in tqdm(
        train_dev_dict.keys(), desc=f"Generating {n}-gram Language Models"
    ):
        lm = generate_n_gram_model(train_dev_dict[author]["train"], n, model_class)
        models.append((author, lm))
        print(
            f"\n{model_class.__name__} Model for {author} has vocab length of {len(lm.vocab)}"
        )
    return models


def parse_args() -> argparse.Namespace:
    """Parse the command line input and return the arguments object."""
    # Code is from ChatGPT
    parser = argparse.ArgumentParser(description="A program to classify authors")

    # Add arguments
    parser.add_argument("authorlist", help="The file containing the list of authors")
    parser.add_argument(
        "-test", dest="testfile", help="The file containing the test data"
    )

    # Parse the arguments
    args = parser.parse_args()
    return args


def train_dev_split(
    authors_dict: Dict[str, List[List[str]]]
) -> Dict[str, Dict[str, List[List[str]]]]:
    """Split our parsed file dictionary into 90% train and 10% dev.
    Example of an entry in the return dict is as follows:\n
    {'austen': {'train': train_lines, 'dev': dev_lines}}.
    To access Austen's train set, do: train_dev_dict['austen']['train']"""

    print("\nSplitting data into training and development sets...")
    # code from ChatGPT
    # Split each list into train and dev sets
    train_dev_dict = {}
    for author, lines in authors_dict.items():
        # Shuffle the lines to ensure randomness
        random.shuffle(lines)
        # Split the lines into train and dev sets
        dev_size = int(0.1 * len(lines))
        dev_lines = lines[:dev_size]
        train_lines = lines[dev_size:]
        # Add the train and dev sets to the new dictionary
        train_dev_dict[author] = {"train": train_lines, "dev": dev_lines}

    # Make sure the counts are correct for the train / dev split.
    for author in train_dev_dict:
        train_len = len(train_dev_dict[author]["train"])
        dev_len = len(train_dev_dict[author]["dev"])

        print(
            f"{author} train length: {train_len}, dev: {dev_len}, (dev_len / train_len): {dev_len / train_len}"
        )

    return train_dev_dict


def main():
    # args.authorlist is required, args.testfile is optional (may be None).
    args = parse_args()

    # parse all the files and train/dev split as necessary.
    authors_dict = tokenize_files(args.authorlist)
    if args.testfile is None:
        train_dev_dict = train_dev_split(authors_dict)

        # Let's generate some language models.
        # Each item in this list is a tuple of (authorname: str, model: MLE)
        # bigram_mle_models = generate_list_of_models(train_dev_dict, 2, MLE)
        bigram_stupidbackoff_models = generate_list_of_models(train_dev_dict, 2, StupidBackoff)

    # The test flag exists. Use ALL of the lines for each author as training data (no dev data).
    else:
        encoding = "utf8" if "utf8" in args.testfile else "ascii"

        # I don't know what our test file is supposed to be, but I'll assume it's not a problem for now.
        # TODO there's a process_test_file function in file_tokenizer.py that needs to be done ig
        test_data = process_test_file(args.testfile, encoding)

    # TODO Use smoothing / backoff / interpolation to see which has the best performance
    # test different values of "n" in n-gram. We can record these in a spreadsheet and make graphs / charts.


if __name__ == "__main__":
    main()
