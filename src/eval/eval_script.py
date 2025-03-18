import pandas as pd

# load answers from CSV file
answers = pd.read_csv('eval.csv', delimiter=";", header=None)

# path to the evaluation answers file
solution = 'eval_answers.txt'

# read evaluation answers file
with open(solution, 'r', encoding='utf-8') as file1:
    Lines = file1.readlines()

# Counter for matches found
a = 0

# iterate through each line in the evaluation answers file
for i, line in enumerate(Lines):
    match_found = False
    # iterate through each item in the answers dataframe for the current row
    for item in answers.iloc[i]:
        if str(item) in line:
            # match found, print the line number and matched item
            print(f"Match found in line {i + 1} of text file: {item}")
            a += 1
            match_found = True
            break
    # if no match found for the current line, print a message
    if not match_found:
        print(f"No match for line {i + 1}")

# calculate and print the percentage of matches found
print((a/(i + 1))*100)
