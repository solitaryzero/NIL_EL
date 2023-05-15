import argparse
import json
import csv
import os


def main(args):
    nil_types = {}
    with open(args.nil_type_path, 'r', encoding='utf-8') as fin:
        reader = csv.reader(fin)
        header = next(reader)
        for row in reader:
            eid = int(row[0])
            source = row[1]
            left_context = row[2]
            mention = row[3]
            right_context = row[4]
            nil_type = row[5]

            if (source == 'test'):
                nil_types[eid] = (nil_type, mention)

    results = {
        'Missing Entity': {
            'correct': 0, 
            'wrong': 0, 
            'total': 0, 
        },
        'Non-Entity Phrase': {
            'correct': 0, 
            'wrong': 0, 
            'total': 0, 
        },
    }

    full_prediction_path = os.path.join(args.prediction_path, args.model, 'NEL_prediction.txt')
    with open(full_prediction_path, 'r', encoding='utf-8') as fin:
        for i, line in enumerate(fin):
            js = json.loads(line)
            if (i in nil_types):
                assert js['mention'] == nil_types[i][1]
                assert js['answer'] == 'NIL'

                if (js['answer'] == js['prediction']):
                    results[nil_types[i][0]]['correct'] += 1
                else:
                    results[nil_types[i][0]]['wrong'] += 1

                results[nil_types[i][0]]['total'] += 1

    print('Results of %s model' %(args.model))
    print(json.dumps(results, indent=4))

    with open(os.path.join(args.output_path, '%s_nil_results.txt' %args.model), 'w', encoding='utf-8' ) as fout:
        json.dump(results, fout, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_path', type=str, default='../data/prediction')
    parser.add_argument('--nil_type_path', type=str, default='../benchmark/NEL/nil_type.csv')
    parser.add_argument('--output_path', type=str, default='../data/nil_evaluation')
    parser.add_argument('--model', type=str, required=True)

    args = parser.parse_args()
    main(args)