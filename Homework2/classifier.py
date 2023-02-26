import argparse
import random
from typing import Dict, List, Type
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm.models import MLE
from file_tokenizer import tokenize_files
from tqdm import tqdm

random.seed(1234)


def generate_n_gram_model(text: List[List[str]], n: int) -> MLE:
    """Returns an MLE n-gram model which has been trained on the text provided.
    Text should be a list of split sentences."""
    train, vocab = padded_everygram_pipeline(n, text)

    # Maximum Likelihood Estimation based model.
    lm = MLE(n)
    lm.fit(train, vocab)
    return lm


def parse_args() -> argparse.Namespace:
    """Parse the command line input and return the arguments object."""
    # Code is from ChatGPT
    parser = argparse.ArgumentParser(
        description='A program to classify authors')

    # Add arguments
    parser.add_argument(
        'authorlist', help='The file containing the list of authors')
    parser.add_argument('-test', dest='testfile',
                        help='The file containing the test data')

    # Parse the arguments
    args = parser.parse_args()
    return args


def train_dev_split(authors_dict: Dict[str, List[str]]) -> Dict[str, Dict[str, List[str]]]:
    """Split our parsed file dictionary into 90% train and 10% dev. Example of an entry is as follows:\n
    {'Austen': {'train': train_lines, 'dev': dev_lines}}.
    To access Austen's train set, do: train_dev_dict['Austen']['train']"""
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
        train_dev_dict[author] = {'train': train_lines, 'dev': dev_lines}

    # Make sure the counts are correct for the train / dev split.
    for author in train_dev_dict:
        train_len = len(train_dev_dict[author]['train'])
        dev_len = len(train_dev_dict[author]['dev'])

        print(
            f'{author} train length: {train_len}, dev: {dev_len}, (dev_len / train_len): {dev_len / train_len}'
        )

    return train_dev_dict


def main():
    # args.authorlist is required, args.testfile is optional (may be None).
    args = parse_args()

    # parse all the files and train/dev split as necessary.
    authors_dict = tokenize_files(args.authorlist)
    if args.testfile is None:
        train_dev_dict = train_dev_split(authors_dict)

        # Generate all our language models.
        # Each item in this list is a tuple of (authorname: str, model: MLE)
        language_models = []
        for author in tqdm(train_dev_dict.keys(), desc="\nGenerating Language Models"):
            lm = generate_n_gram_model(train_dev_dict[author]['train'], 2)
            language_models.append((author, lm))
            print(f'\nModel for {author} has vocab length of {len(lm.vocab)}')

    # The test flag exists. Use ALL of the lines for each author as training data (no dev data).
    else:
        # TODO
        pass

    # Use smoothing / backoff / interpolation to see which has the best performance
    # test different values of "n" in n-gram


if __name__ == "__main__":
    main()
