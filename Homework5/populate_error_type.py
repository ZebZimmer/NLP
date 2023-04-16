import csv

input_file = 'CSCI_5541_spacy_annotated.csv'
output_file = 'new_CSCI_5541_spacy_annotated.csv'

with open(input_file, 'r', newline='', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        ground_truth = row['Ground-Truth Label']
        predicted_label = row['Predicted Label'].split()[-1]  # Extract the label from the "Predicted Label" column (ignores "Confidence: x.xxxx")
        
        if ground_truth != predicted_label:
            row['Error Type'] = f"False {predicted_label}"
        else:
            row['Error Type'] = 'Correct'
        
        writer.writerow(row)
