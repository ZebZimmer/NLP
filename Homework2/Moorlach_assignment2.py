import nltk
import os


AUSTEN = 0
DICKENS = 1
TOLSTOY = 2
WILDE = 3

ascii = True
utf8 = False

authorfile = open("authorlist.txt", "r")

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

print(authorList)
print("ASCII: ", ascii)
print("UTF8: ", utf8)
authorfile.close()

austenLines = []
dickensLines = []
tolstoyLines = []
wildeLines = []

austenString = ""
dickensString = ""
tolstoyString = ""
wildeString = ""

nltk.download('punkt')
print()

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


if(ascii):
    #Encoding is in ASCII

    if(authorList[AUSTEN]):
        #We need the Austen Lines

        austenfile = open("ngram_authorship_train/austen.txt", "r")
        data = austenfile.read().splitlines()
        for line in data:
            if(len(line) != 0):
                if(len(austenString) == 0):
                    austenString = line
                else:
                    austenString += " " + line
        austenLines = tokenizer.tokenize(austenString)

    if(authorList[DICKENS]):
        #We need the dickens Lines

        dickensfile = open("ngram_authorship_train/dickens.txt", "r")
        data = dickensfile.read().splitlines()
        for line in data:
            if(len(line) != 0):
                if(len(dickensString) == 0):
                    dickensString = line
                else:
                    dickensString += " " + line
        dickensLines = tokenizer.tokenize(dickensString)
    if(authorList[TOLSTOY]):
        #We need the tolstoy Lines

        tolstoyfile = open("ngram_authorship_train/tolstoy.txt", "r")
        data = tolstoyfile.read().splitlines()
        for line in data:
            if(len(line) != 0):
                if(len(tolstoyString) == 0):
                    tolstoyString = line
                else:
                    tolstoyString += " " + line
        tolstoyLines = tokenizer.tokenize(tolstoyString)
    if(authorList[WILDE]):
        #We need the wilde Lines

        wildefile = open("ngram_authorship_train/wilde.txt", "r")
        data = wildefile.read().splitlines()
        for line in data:
            if(len(line) != 0):
                if(len(wildeString) == 0):
                    wildeString = line
                else:
                    wildeString += " " + line
        wildeLines = tokenizer.tokenize(wildeString)
      
   

if(utf8):
    #Encoding is in ASCII

    if(authorList[AUSTEN]):
        #We need the Austen Lines

        austenfile = open("ngram_authorship_train/austen.txt", "r")
        data = austenfile.read().splitlines()
        for line in data:
            if(len(line) != 0):
                if(len(austenString) == 0):
                    austenString = line
                else:
                    austenString += " " + line
        austenLines = tokenizer.tokenize(austenString)

    if(authorList[DICKENS]):
        #We need the dickens Lines

        dickensfile = open("ngram_authorship_train/dickens.txt", "r")
        data = dickensfile.read().splitlines()
        for line in data:
            if(len(line) != 0):
                if(len(dickensString) == 0):
                    dickensString = line
                else:
                    dickensString += " " + line
        dickensLines = tokenizer.tokenize(dickensString)
    if(authorList[TOLSTOY]):
        #We need the tolstoy Lines

        tolstoyfile = open("ngram_authorship_train/tolstoy.txt", "r")
        data = tolstoyfile.read().splitlines()
        for line in data:
            if(len(line) != 0):
                if(len(tolstoyString) == 0):
                    tolstoyString = line
                else:
                    tolstoyString += " " + line
        tolstoyLines = tokenizer.tokenize(tolstoyString)
    if(authorList[WILDE]):
        #We need the wilde Lines

        wildefile = open("ngram_authorship_train/wilde.txt", "r")
        data = wildefile.read().splitlines()
        for line in data:
            if(len(line) != 0):
                if(len(wildeString) == 0):
                    wildeString = line
                else:
                    wildeString += " " + line
        wildeLines = tokenizer.tokenize(wildeString)
