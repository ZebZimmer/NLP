import argparse
import random
from typing import Dict, List, Tuple
from nltk import pad_sequence
from nltk.lm import Vocabulary
from nltk.util import bigrams, ngrams, everygrams
from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends
from nltk.lm.models import MLE, LanguageModel, Laplace
from file_tokenizer import tokenize_files, parse_file, process_test_file
from tqdm import tqdm

random.seed(1234)
NGRAM = 2


def generate_n_gram_model(
    text: List[List[str]], n: int, model_type: LanguageModel
) -> LanguageModel:
    """Returns an n-gram model which has been trained on the text provided.
    Text should be a list of split sentences.
    model_type is an nltk language model class which has a supports .fit()."""
    train, vocab = padded_everygram_pipeline(n, text)
    lm = model_type(n)
    lm.fit(train, vocab)
    return lm


# function from ChatGPT
def generate_list_of_models(
    train_dev_dict: Dict[str, Dict[str, List[List[str]]]],
    n: int,
    model_class: LanguageModel,
) -> List[Tuple[str, LanguageModel]]:
    """Return a list of Tuples. Tuple[0] is a string for the author name, like 'austen'.
    Tuple[1] is a trained language model based on each author's respective train data."""
    print()
    models = []
    for author in tqdm(
        train_dev_dict.keys(), desc=f"Generating {n}-gram Language Models"
    ):
        lm = generate_n_gram_model(train_dev_dict[author]["train"], n, model_class)
        models.append((author, lm))
        print(f"\n{model_class.__name__} Model for {author} has vocab length of {len(lm.vocab)}")
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

def hamzas_func(line, model, n):
    temp = []
    test_ngram = list(everygrams(pad_sequence(line,n, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'), max_len=n))
    for ngram in test_ngram:
        if model.perplexity([ngram]) !=  float('inf'):
            temp.append(ngram)
    if(len(temp) == 0):
        # print("we removed all ngrams")
        # print(test_ngram)
        return []
    test_ngram = temp


    return test_ngram

def test_dev_set(
        bigram_mle_models: List[Tuple[str, LanguageModel]],
        train_dev_dict: Dict[str, Dict[str, List[List[str]]]],
        n: int
):
    """Test each sentence in the development set of the train_dev_dict.
    We should find the perplexity of that sentence in each language model
    and pick the lowest one.  Then we will need to print out the accuracy scores"""


    # print(austenLine)
    correctNames = []
    for model in bigram_mle_models:
        correctNames.append(model[0])


    totalLines = 0
    correctLines = 0
    # correctNames = ["austen","dickens","tolstoy","wilde"]
    for correctName in correctNames:

        for line in train_dev_dict[correctName]["dev"]:
            totalLines += 1
            # print("the line is: ", line)
            lowest = float('inf')
            lowest_name = "NA"
            for model in bigram_mle_models:
                name = model[0]
                # print(name)
                model = model[1]
                # temp = []
                # test_ngram = list(everygrams(pad_sequence(line,n, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'), max_len=n))
                # for ngram in test_ngram:
                #     if model.perplexity([ngram]) !=  float('inf'):
                #         temp.append(ngram)
                # if(len(temp) == 0):
                #     # print("we removed all ngrams")
                #     # print(test_ngram)
                #     continue
                # test_ngram = temp
                test_ngram = hamzas_func(line, model, n)
                if(len(test_ngram)==0):
                    print("test_ngram empty")
                    continue
                value = model.perplexity(test_ngram)
                if(value < lowest):
                    lowest = value
                    lowest_name = name
            if(lowest_name == correctName):
                correctLines += 1
            # print(lowest_name)
        print(correctName, " accuracy: ", correctLines/totalLines)
    return

def main():
    # args.authorlist is required, args.testfile is optional (may be None).
    args = parse_args()

    # parse all the files and train/dev split as necessary.
    authors_dict = tokenize_files(args.authorlist)
    # print(authors_dict['austen'][0])
    if args.testfile is None:
        train_dev_dict = train_dev_split(authors_dict)

        # print(train_dev_dict["austen"]["dev"][1])

        # Let's generate some MLE bigram language models.
        # Each item in this list is a tuple of (authorname: str, model: MLE)
        bigram_mle_models = generate_list_of_models(train_dev_dict, NGRAM, Laplace)
        test_dev_set(bigram_mle_models, train_dev_dict, NGRAM)



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
