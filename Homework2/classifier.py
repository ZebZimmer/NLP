import argparse
import random
import nltk
import numpy as np
from typing import Dict, List, Tuple
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm.models import (
    MLE,
    LanguageModel,
    StupidBackoff,
    Lidstone,
    KneserNeyInterpolated,
    Laplace
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
        train_dev_dict.keys(), desc=f"Generating {n}-gram {model_class.__name__} Language Models"
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

def evaluate_models(
    train_dev_dict: Dict[str, Dict[str, List[List[str]]]],
    model_list: List[Tuple[str, LanguageModel]],
) -> None:
    """Evaluate the models on the development set and print the results"""
    # function from ChatGPT
    print("Results on dev set:")
    for author in train_dev_dict.keys():
        dev_data = train_dev_dict[author]["dev"]
        correct_count = 0
        total_count = 0

        for sentence in dev_data:
            total_count += 1
            predicted_author = get_best_perplexity(sentence, model_list)
            if predicted_author == author:
                correct_count += 1

        accuracy = correct_count / total_count * 100
        print(f"{author} {accuracy:.1f}% correct")
        
    print('\n')


def get_best_perplexity(
    test_record: List[str], model_list: List[Tuple[str, LanguageModel]]
) -> str:
    """Returns the author's name of the model which has the lowest perplexity on test_record.\n
    test_record is just a list of strings like ['I', 'Am', 'Cool']\n
    model_list is a list of (authorname, model) tuples, likely created by generate_list_of_models(...)
    This function assumes all models in model_list are the same order.
    """
    # received ChatGPT help to complete this.
    # Convert test record into a sequence of n-grams
    # this includes all n-grams up to and including n.
    n = model_list[0][1].order
    text_ngrams = list(nltk.everygrams(test_record, max_len=n))

    # run through all authors in model_list and get lowest perplexity author.
    best_author = model_list[0][0]
    best_perplexity = np.inf

    for author, lm in model_list:
        current_model_perplexity = lm.perplexity(text_ngrams)
        if current_model_perplexity == np.inf:
            print(f'GOT NP.INF! {author}, {text_ngrams}')
        if current_model_perplexity < best_perplexity:
            best_perplexity = current_model_perplexity
            best_author = author

    return best_author


def main():
    # args.authorlist is required, args.testfile is optional (may be None).
    args = parse_args()

    # parse all the files and train/dev split as necessary.
    authors_dict = tokenize_files(args.authorlist)
    if args.testfile is None:
        train_dev_dict = train_dev_split(authors_dict)

        bigram_laplace_models = generate_list_of_models(train_dev_dict, 2, Laplace)
        print('\nTesting Laplace models, ', end='')
        evaluate_models(train_dev_dict, bigram_laplace_models)

        bigram_lidstone_models = generate_list_of_models(train_dev_dict, 2, Lidstone)
        print('\nTesting Lidstone models, ', end='')
        evaluate_models(train_dev_dict, bigram_lidstone_models)

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
