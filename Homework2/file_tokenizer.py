import nltk
import os
from typing import Dict, List

AUSTEN = 0
DICKENS = 1
TOLSTOY = 2
WILDE = 3

nltk.download('punkt')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def parse_file(filename: str) -> List[str]:
    """Parse a txt file and return a list of strings."""
    s = ''
    f = open(filename, "r")
    data = f.read().splitlines()
    for line in data:
        if(len(line) != 0):
            if(len(s) == 0):
                s = line
            else:
                s += " " + line
    return tokenizer.tokenize(s)

def tokenize_files(authorlistFilename: str) -> Dict[str, List[str]]:
    """Parse the local authorlist.txt file into lists of sentences.
    Returns a dictionary as follows:"""
    ascii = True
    utf8 = False
    authorfile = open(authorlistFilename, "r")
    authorList = [False, False, False, False]
    authorFileList = authorfile.read().splitlines()
    print(authorFileList)

    for line in authorFileList:
        if("#" in line):    #ignore commented out lines in the author list file
            continue
        if("utf8" in line): #switch to utf8 encoding
            ascii = False
            utf8 = True
        if("austen" in line): 
            authorList[AUSTEN] = True
        if("dickens" in line):
            authorList[DICKENS] = True
        if("tolstoy" in line):
            authorList[TOLSTOY] = True
        if("wilde" in line):
            authorList[WILDE] = True

    print(f'authorList: {authorList}')
    print("ASCII: ", ascii)
    print("UTF8: ", utf8)
    authorfile.close()

    austenLines = []
    dickensLines = []
    tolstoyLines = []
    wildeLines = []

    if(authorList[AUSTEN]):
        #We need the Austen Lines
        austenLines = parse_file("ngram_authorship_train/austen.txt")

    if(authorList[DICKENS]):
        #We need the dickens Lines
        dickensLines = parse_file("ngram_authorship_train/dickens.txt")

    if(authorList[TOLSTOY]):
        #We need the tolstoy Lines
        tolstoyLines = parse_file("ngram_authorship_train/tolstoy.txt")
        
    if(authorList[WILDE]):
        #We need the wilde Lines
        wildeLines = parse_file("ngram_authorship_train/wilde.txt")

    encoding = 'utf8' if utf8 else 'ascii'
    print(f'Used {encoding} to tokenize files')
    print(f'lens: wildeLines - {len(wildeLines)}, austen: {len(austenLines)}, tolstoy: {len(tolstoyLines)}, dickens: {len(dickensLines)}')
    print(f'wildeLines[0]: {wildeLines[0]}')

    # ChatGPT code below to generate authors_dict (our return value).
    authors_dict = {
        'Austen': austenLines,
        'Dickens': dickensLines,
        'Tolstoy': tolstoyLines,
        'Wilde': wildeLines,
    }

    authors_dict = {k: v for k, v in authors_dict.items() if v}
    return authors_dict