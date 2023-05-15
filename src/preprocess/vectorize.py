import json
import argparse
import os
import random
import csv

def work(args):
    # read KB
    type_id_map = {}
    kb = {}
    kb_path = os.path.join(args.data_path, 'kb.jsonl')
    with open(kb_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            js = json.loads(line)
            title = js['title']
            abstract = js['abstract']
            types = js['type']

            if (args.single_type) and (len(types) == 0):
                types = ['Other']

            for typ in types:
                if (typ not in type_id_map):
                    type_id_map[typ] = len(type_id_map)

            if (args.single_type):
                types = [types[0]]
            kb[title] = (abstract, types)

    if not(os.path.exists(args.out_path)):
        os.makedirs(args.out_path)
    if not(os.path.exists(args.benchmark_out_path)):
        os.makedirs(args.benchmark_out_path)

    # read type map
    type_map_path = os.path.join(args.out_path, 'type_map.jsonl')
    with open(type_map_path, 'w', encoding='utf-8') as fout:
        for typ in type_id_map:
            js = {
                'type_name': typ,
                'id': type_id_map[typ],
            }
            fout.write(json.dumps(js))
            fout.write('\n')

    # read Non-Entity Phrase types
    nep_examples = set()
    with open(args.nil_type_path, 'r', encoding='utf-8') as fin:
        reader = csv.reader(fin)
        header = next(reader)
        for row in reader:
            eid = row[0]
            source = row[1]
            left_context = row[2]
            mention = row[3]
            right_context = row[4]
            nil_type = row[5]

            if (nil_type.lower() == 'Non-Entity Phrase'.lower()):
                nep_examples.add((int(eid), source))
                

    single_splits = ['train', 'valid', 'test']
    for spl in single_splits:
        json_path = os.path.join(args.data_path, '%s.jsonl' %spl)

        null_abstracts = set()
        all_data = []

        if (args.partial_nil):
            nil_indexes = []
            non_nil_indexes = set()

        if (args.partial_nep):
            nep_indexes = []
            non_nep_indexes = set()

        with open(json_path, 'r', encoding='utf-8') as fin:
            for i, line in enumerate(fin):
                js = json.loads(line)
                mention = js['mention']
                left_context = js['left_context']
                right_context = js['right_context']
                answer = js['answer']
                candidates = js['candidate_title']

                if (answer != 'NIL'):
                    context_type = kb.get(answer, ('', []))[1]
                    context_type_ids = [0]*len(type_id_map)
                    for t in context_type:
                        context_type_ids[type_id_map[t]] = 1
                else:
                    context_type = None
                    context_type_ids = [0]*len(type_id_map)
                    if (args.partial_nil):
                        nil_indexes.append([])
                    if (args.partial_nep):
                        nep_indexes.append([])

                for cand in candidates:
                    if (cand == answer):
                        flag = 1
                    else:
                        flag = 0

                    abstract, types = kb.get(cand, ('', None))
                    type_ids = [0]*len(type_id_map)
                    if (types is not None):
                        for t in types:
                            type_ids[type_id_map[t]] = 1

                    if (abstract == ''):
                        null_abstracts.add(cand)
                    data = {
                        'mention': mention,
                        'left_context': left_context,
                        'right_context': right_context,
                        'context_type': context_type,
                        'context_type_ids': context_type_ids,
                        'entity_title': cand,
                        'entity_abstract': abstract,
                        'entity_type': types,
                        'entity_type_ids': type_ids,
                        'label': flag,
                    }

                    if (args.partial_nil):
                        if (answer == 'NIL'):
                            nil_indexes[-1].append(len(all_data))
                        else:
                            non_nil_indexes.add(len(all_data))

                    if (args.partial_nep):
                        if (answer == 'NIL') and ((i, spl) in nep_examples):
                            nep_indexes[-1].append(len(all_data))
                        else:
                            non_nep_indexes.add(len(all_data))

                    all_data.append(data)

        if (args.partial_nil):
            # downsample NIL examples
            retain_num = int(len(nil_indexes)*args.nil_percentage)
            _nil_indexes = random.sample(nil_indexes, retain_num)
            sampled_nil_indexes = set()
            for _ in _nil_indexes:
                for ind in _:
                    sampled_nil_indexes.add(ind)

            sampled_data = []
            for i, data in enumerate(all_data):
                if (i in sampled_nil_indexes) or (i in non_nil_indexes):
                    sampled_data.append(data)

            all_data = sampled_data
        elif (args.partial_nep):
            # downsample Non-Entity Phrase examples
            retain_num = int(len(nep_indexes)*args.nep_percentage)
            _nep_indexes = random.sample(nep_indexes, retain_num)
            sampled_nep_indexes = set()
            for _ in _nep_indexes:
                for ind in _:
                    sampled_nep_indexes.add(ind)

            sampled_data = []
            for i, data in enumerate(all_data):
                if (i in sampled_nep_indexes) or (i in non_nep_indexes):
                    sampled_data.append(data)

            all_data = sampled_data

        print('Entity without abstract: %d' %len(null_abstracts))
        if not(os.path.exists(args.out_path)):
            os.makedirs(args.out_path)

        out_path = os.path.join(args.out_path, '%s.jsonl' %spl)
        with open(out_path, 'w', encoding='utf-8') as fout:
            for data in all_data:
                fout.write(json.dumps(data, ensure_ascii=False))
                fout.write('\n')

        null_path = os.path.join(args.out_path, '%s_missing_entities.txt' %spl)
        with open(null_path, 'w', encoding='utf-8') as fout:
            for ent in null_abstracts:
                fout.write(ent)
                fout.write('\n')

    multi_splits = ['train', 'valid', 'test']
    for spl in multi_splits:
        json_path = os.path.join(args.data_path, '%s.jsonl' %spl)

        null_abstracts = set()
        all_data = []
        with open(json_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                js = json.loads(line)
                mention = js['mention']
                left_context = js['left_context']
                right_context = js['right_context']
                answer = js['answer']
                candidates = js['candidate_title']

                if (answer != 'NIL'):
                    context_type = kb.get(answer, ('', []))[1]
                    context_type_ids = [0]*len(type_id_map)
                    for t in context_type:
                        context_type_ids[type_id_map[t]] = 1
                else:
                    context_type = None
                    context_type_ids = [0]*len(type_id_map)

                candidate_abstracts = []
                candidate_types = []
                candidate_type_ids = []
                labels = []

                for cand in candidates:
                    if (cand == answer):
                        flag = 1
                    else:
                        flag = 0

                    abstract, types = kb.get(cand, ('', None))
                    type_ids = [0]*len(type_id_map)
                    if (types is not None):
                        for t in types:
                            type_ids[type_id_map[t]] = 1

                    if (abstract == ''):
                        null_abstracts.add(cand)
                    candidate_abstracts.append(abstract)
                    labels.append(flag)
                    candidate_types.append(types)
                    candidate_type_ids.append(type_ids)

                data = {
                    'mention': mention,
                    'left_context': left_context,
                    'right_context': right_context,
                    'context_type': context_type,
                    'context_type_ids': context_type_ids,
                    'entity_title': candidates,
                    'entity_abstract': candidate_abstracts,
                    'entity_type': candidate_types,
                    'entity_type_ids': candidate_type_ids,
                    'label': labels,
                }

                all_data.append(data)

        print('Entity without abstract: %d' %len(null_abstracts))
        if not(os.path.exists(args.benchmark_out_path)):
            os.makedirs(args.benchmark_out_path)

        out_path = os.path.join(args.benchmark_out_path, '%s.jsonl' %spl)
        with open(out_path, 'w', encoding='utf-8') as fout:
            for data in all_data:
                fout.write(json.dumps(data, ensure_ascii=False))
                fout.write('\n')

        null_path = os.path.join(args.out_path, '%s_missing_entities.txt' %spl)
        with open(null_path, 'w', encoding='utf-8') as fout:
            for ent in null_abstracts:
                fout.write(ent)
                fout.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/processed')
    parser.add_argument('--nil_type_path', type=str, default='../data/benchmark/NEL_annotation/NEL_annotated.csv')
    parser.add_argument('--out_path', type=str, default='../data/vector')
    parser.add_argument('--benchmark_out_path', type=str, default='../data/benchmark/NEL')
    parser.add_argument('--single_type', action='store_true')
    parser.add_argument('--partial_nil', action='store_true')
    parser.add_argument('--partial_nep', action='store_true')
    parser.add_argument('--nep_percentage', type=float, default=1.0)
    parser.add_argument('--nil_percentage', type=float, default=1.0)
    args = parser.parse_args()

    work(args)