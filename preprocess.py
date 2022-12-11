import os

def preprocess():
    # removing the new line characters
    with open('data/train 2.txt') as f:
        lines = [line.rstrip() for line in f]
        
    new_lines = []
    for line in lines:
        parts = line.split(' ')
        if len(parts) > 1 and parts[0] == '':
            continue
        new_lines.append(line)

    with open('data/train3.txt', 'w') as f:
        for line in new_lines:
            f.write(f"{line}\n")