import csv
import random
import json
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--error_examples', type=int, default=0)
    args = parser.parse_args()

    file_path = '../data/AIDA_annotation/AIDA_annotated.csv'
    me_num, ne_num, total_num = 0, 0, 0
    error_num = 0
    error_examples = []

    with open(file_path, 'r', encoding='utf-8') as fin:
        reader = csv.reader(fin)
        header = next(reader)
        for row in reader:
            eid = row[0]
            source = row[1]
            left_context = row[2]
            mention = row[3]
            right_context = row[4]
            nil_type = row[5]

            total_num += 1
            if (nil_type.lower() == 'Missing Entity'.lower()):
                me_num += 1
            elif (nil_type.lower() == 'Non-Entity Phrase'.lower()):
                ne_num += 1
            else:
                try:
                    assert nil_type.startswith('Not NIL') 
                except:
                    print(nil_type)
                error_num += 1
                error_examples.append({
                    'left_context': left_context[-100:],
                    'mention': mention,
                    'right_context': right_context[:100],
                    'answer': nil_type[len('Not NIL')+2:-1]
                })

    print('Missing Entity: %d\n Non-Entity Phrase: %d\n Error: %d\n Total: %d'
    %(me_num, ne_num, error_num, total_num))

    if (args.error_examples > 0):
        examples = random.sample(error_examples, args.error_examples)
        for ex in examples:
            print(json.dumps(ex, indent=4))