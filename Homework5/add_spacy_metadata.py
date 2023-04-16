import csv
import spacy

# simply run this script with hw5_original.csv in the same directory and 
# it'll output a new csv with extra spacy columns
# Chat-GPT4 help with code below:
# make sure you have this downloaded: python -m spacy download en_core_web_md
nlp = spacy.load('en_core_web_md')

def add_features(row):
    doc = nlp(row['Input Text'])
    row['Number of Sentences'] = len(list(doc.sents))
    row['Number of Named Entities'] = len(list(doc.ents))
    # row['Sentiment Score'] = doc.sentiment # model does not support sentiment so it's always 0
    row['POS Tags'] = {token.pos_: sum(1 for _ in filter(lambda t: t.pos_ == token.pos_, doc)) for token in doc}
    return row

input_file = 'hw5_original.csv'
output_file = 'hw5_spacy_annotated.csv'

with open(input_file, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    fieldnames = reader.fieldnames + ['(Spacy) Number of Sentences', '(Spacy) Number of Named Entities', '(Spacy) POS Tags']

    with open(output_file, 'w', newline='') as output_csv:
        writer = csv.DictWriter(output_csv, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in reader:
            updated_row = add_features(row)
            writer.writerow(updated_row)