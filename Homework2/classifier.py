import numpy as np
import argparse, random, os
from typing import Dict, List, Tuple
import nltk
from nltk import pad_sequence
from nltk.util import everygrams
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm.models import (
    MLE, 
    LanguageModel, 
    Laplace, 
    Lidstone,
    StupidBackoff,
    WittenBellInterpolated
)
# from nltk.lm import StupidBackoff
from file_tokenizer import tokenize_files, process_sentence, generate_testfile
from tqdm import tqdm

random.seed(1234)
NGRAM = 2


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
        train_dev_dict.keys(),
        desc=f"Generating {n}-gram {model_class.__name__} Language Models",
    ):
        lm = generate_n_gram_model(
            train_dev_dict[author]["train"], n, model_class, alpha, gamma
        )
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


def get_valid_ngrams(
    line: List[str], model: LanguageModel, n: int
) -> List[Tuple[str, ...]]:
    """Given a line, return a list of all valid test_ngrams (of size up to and including n).
    The model must already have been trained using .fit(trian, vocab).
    Returns a list of ngrams. This is ready to be used as such: model.perplexity(test_ngram)
    """
    valid_ngrams = []
    test_ngram = list(
        everygrams(
            pad_sequence(
                line,
                n,
                pad_left=True,
                pad_right=True,
                left_pad_symbol="<s>",
                right_pad_symbol="</s>",
            ),
            max_len=n,
        )
    )
    for ngram in test_ngram:
        if model.perplexity([ngram]) != float("inf"):
            valid_ngrams.append(ngram)
        # else:
        #     valid_ngrams.append(ngram)

    return valid_ngrams


def test_dev_set(
    train_dev_dict: Dict[str, Dict[str, List[List[str]]]],
    model_list: List[Tuple[str, LanguageModel]],
) -> None:
    """Test each sentence in the development set of the train_dev_dict.
    We should find the perplexity of that sentence in each language model
    and pick the lowest one.  Then we will need to print out the accuracy scores"""
    print("Results on dev set:")
    # total_lines = 0
    # correct_lines = 0
    author_names = [model[0] for model in model_list]
    for author in author_names:
        total_lines = 0
        correct_lines = 0
        for line in train_dev_dict[author]["dev"]:
            total_lines += 1
            best_author = get_best_perplexity(line, model_list)

            if not best_author:
                best_author = random.choice(author_names)

            if best_author == author:
                correct_lines += 1

        accuracy = correct_lines / total_lines * 100
        # print(f'correct_lines: {correct_lines}, total_lines: {total_lines}')
        print(f"{author} {accuracy:.1f}% correct")


def evaluate_models(
    train_dev_dict: Dict[str, Dict[str, List[List[str]]]],
    model_list: List[Tuple[str, LanguageModel]],
) -> None:
    """Evaluate the models on the development set and print the results."""
    # function made with help from ChatGPT
    model_type = type(model_list[0][1]).__name__
    print(f"\nTesting {model_type} models, Results on respective dev sets:")

    author_names = [model[0] for model in model_list]

    # use these variables to compute the overall accuracy of all models across all dev sets.
    total_lines, total_correct = 0, 0
    for author in author_names:
        dev_data = train_dev_dict[author]["dev"]

        # we'll compute an author's accuracy on their respective dev set.
        dev_set_count = len(dev_data)
        dev_correct_count = 0
        for sentence in dev_data:
            total_lines += 1
            predicted_author = get_best_perplexity(sentence, model_list)
            if predicted_author == author:
                total_correct += 1
                dev_correct_count += 1

        accuracy = dev_correct_count / dev_set_count * 100
        print(f"{author} {accuracy:.1f}% correct")

    overall_accuracy = total_correct / total_lines * 100
    print(
        f"Overall accuracy across all models and dev sets: {overall_accuracy:.1f}% correct"
    )


def get_best_perplexity(
    test_record: List[str], model_list: List[Tuple[str, LanguageModel]]
) -> str:
    """Returns the author's name of the model which has the lowest perplexity on test_record.\n
    test_record is just a list of strings like ['I', 'Am', 'Cool']\n
    model_list is a list of (authorname, model) tuples, likely created by generate_list_of_models(...)
    This function assumes all models in model_list are the same n-gram order.
    """
    # received ChatGPT help to complete this.
    n = model_list[0][1].order

    # run through all authors in model_list and get lowest perplexity author.
    best_author = None
    best_perplexity = np.inf
    for author, lm in model_list:
        test_ngrams = get_valid_ngrams(test_record, lm, n)
        if not test_ngrams:
            continue

        current_model_perplexity = lm.perplexity(test_ngrams)
        if current_model_perplexity < best_perplexity:
            best_perplexity = current_model_perplexity
            best_author = author

    return best_author


def testfile_evaluation(
    filename: str,
    authors_dict: Dict[str, List[List[str]]],
    generated: bool = False,
    model_class: LanguageModel = Laplace,
    n: int = 2,
) -> None:
    """Given a testfile name which is either ascii or utf encoded and a dictionary of author names
    and processed sentences. For each sentence in the testfile, the most likely author is printed to the terminal.\n
    Param generated: Specifies if we created the test file or not. Alters testfile parsing.
    Param model_class: specifies which type of nltk model will be used for evaluation.\n
    Param n: specifies the order of the n-gram models to be created.
    """
    # create a "train_dict" that only has train data and no dev data.
    # these models will be trained on ALL of our processed file data.
    train_dict = {}
    for author, lines in authors_dict.items():
        train_dict[author] = {"train": lines, "dev": []}

    # generate some n-gram models
    models = generate_list_of_models(train_dict, n, model_class)

    encoding = "utf8" if "utf8" in filename else "ascii"
    total_lines, correct_predictions = 0, 0
    correct_author = None
    print(f"\nEvaluating testfile using {model_class.__name__} models. Classifications:")
    with open(filename, encoding=encoding) as f:
        # each line in the CUSTOM testfile is in the format of f"{sentence}@{correct_author}"
        # this is not the case for the testfile used in the grading script.
        for line in f:
            total_lines += 1

            if generated:
                line, correct_author = line.rstrip("\n").split("@")

            processed_test_record = process_sentence(line, stop=False)
            best_author = get_best_perplexity(processed_test_record, models)

            # print the terminal output
            if correct_author is not None:
                print(f"predicted {best_author}, correct author is {correct_author}")
                if best_author == correct_author:
                    correct_predictions += 1
            
            else:
                print(best_author)

    if correct_author is not None:
        print(
            f"{correct_predictions} correct classifications out of {total_lines} lines. Accuracy: {correct_predictions / total_lines}"
        )

def generate_text(list_of_models, text_seed, number_of_words):
    print()
    print()
    for author, model in list_of_models:
        model_type = type(list_of_models[0][1]).__name__
        print("Generating ", author, " text from the ", model_type, " model")
        for i in range(5):
            text = model.generate(number_of_words, text_seed=text_seed, random_seed=i)
            full = []
            for word in text_seed:
                full.append(word)
            for word in text:
                full.append(word)
            for word in full:
                print(word, end=" ")
            print()

            for author2, model2 in list_of_models:
                ngrams = get_valid_ngrams(full, model2, NGRAM)
                print("The perplexity of that sentence on the ", author2, " model is : ", model2.perplexity(ngrams))
            # ngrams = get_valid_ngrams(full, model, NGRAM)
            # print("perplexity : ", model.perplexity(ngrams))
            print()
        print()
        print()

def find_best(list_of_models, text_seed, number_of_words, number_of_iters):
    print()
    print()
    for author, model in list_of_models:
        model_type = type(list_of_models[0][1]).__name__
        print("Generating best sentence from ", author, " on the ", model_type, " model")
        best_perp = float('inf')
        best_sent = ""
        for i in range(number_of_iters):
            text = model.generate(number_of_words, text_seed=text_seed, random_seed=i)
            full = []
            for word in text_seed:
                full.append(word)
            for word in text:
                full.append(word)
            ngrams = get_valid_ngrams(full, model, NGRAM)
            if model.perplexity(ngrams) < best_perp:
                best_perp = model.perplexity(ngrams)
                best_sent = full
        for word in best_sent:
            print(word, end = " ")
        print()
        print('perplexity: ', best_perp)

            
        print()
        print()



def main():
    # args.authorlist is required, args.testfile is optional (may be None).
    args = parse_args()

    # parse all the files
    authors_dict = tokenize_files(args.authorlist)

    if args.testfile is None:
        train_dev_dict = train_dev_split(authors_dict)

        # Let's generate some ngram language models and evaluate them
        # bigram_mle_models = generate_list_of_models(train_dev_dict, NGRAM, MLE)
        # evaluate_models(train_dev_dict, bigram_mle_models)

        # bigram_backoff_models = generate_list_of_models(train_dev_dict, 2, nltk.lm.StupidBackoff)

        bigram_lidstone_models = generate_list_of_models(train_dev_dict, 2, Lidstone)

        # bigram_interpolated_models = generate_list_of_models(train_dev_dict, 2, WittenBellInterpolated)

        # bigram_MLE_models = generate_list_of_models(train_dev_dict, 2, MLE)

        # bigram_laplace_models = generate_list_of_models(train_dev_dict, 2, Laplace)
        # test_dev_set(train_dev_dict, bigram_laplace_models)
        # evaluate_models(train_dev_dict, bigram_MLE_models)
        # evaluate_models(train_dev_dict, bigram_laplace_models)
        # evaluate_models(train_dev_dict, bigram_interpolated_models)
        # evaluate_models(train_dev_dict, bigram_backoff_models)
        # evaluate_models(train_dev_dict, bigram_backoff_models)
        evaluate_models(train_dev_dict ,bigram_lidstone_models)
        # evaluate_models(train_dev_dict, bigram_laplace_models)
        # generate_text(bigram_laplace_models, ["<s>","i"], 12)
        generate_text(bigram_lidstone_models, ["<s>","i"], 12)
        # find_best(bigram_laplace_models,  ["<s>","i"], 12, 50)
        # find_best(bigram_lidstone_models,  ["<s>","i"], 12, 50)

        # bigram_lidstone_models = generate_list_of_models(train_dev_dict, 2, Lidstone)
        # evaluate_models(train_dev_dict, bigram_lidstone_models)

    else:
        generated = generate_testfile(args.authorlist, args.testfile, num_lines=10)
        testfile_evaluation(
            args.testfile, authors_dict, generated=generated, model_class=Lidstone
        )


if __name__ == "__main__":
    main()
