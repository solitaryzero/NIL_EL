# coding: utf-8

import os
import pickle
from tqdm import tqdm
import argparse
import json
from collections import defaultdict


def process_example(js, 
    redirect_map,
    alias_table,
    abstract_map,
):

    left_content = js['context_left']
    right_content = js['context_right']
    mention = js['mention']
    wiki_id = js['Wikipedia_ID']
    wiki_title = js['Wikipedia_title']
    wiki_url = js['Wikipedia_URL']

    if (wiki_id is None):
        return None

    wiki_id = int(wiki_id)
    if (wiki_id in redirect_map):
        wiki_id = redirect_map[wiki_id]

    candidates = alias_table.get(mention.lower(), None)
    if (candidates is None):
        return None

    label = []
    flag = False
    for cand in candidates:
        if (cand == wiki_title):
            label.append(1)
            flag = True
        else:
            label.append(0)

    if not(flag):
        candidates.append(mention)
        label.append(1)

    abstracts = []
    for cand in candidates:
        abstract = abstract_map.get(cand, '')
        abstracts.append(abstract)

    new_js = {
        'left_context': left_content,
        'right_context': right_content,
        'mention': mention,
        'entity_title': candidates,
        'entity_abstract': abstracts,
        'labels': label,
    }

    return new_js


def main(args):
    dataset_base_paths = {
        'ace2004_questions': 'ace',
        'AIDA-YAGO2_testa': 'AIDA',
        'AIDA-YAGO2_testb': 'AIDA',
        'AIDA-YAGO2_train': 'AIDA',
        'AIDA-YAGO2_testa_nil': 'AIDA_nil',
        'AIDA-YAGO2_testb_nil': 'AIDA_nil',
        'AIDA-YAGO2_train_nil': 'AIDA_nil',
        'aquaint_questions': 'aqua',
        'clueweb_questions': 'clueweb',
        'msnbc_questions': 'msnbc',
        'wnedwiki_questions': 'wned',
    }

    benchmark_path = os.path.join(args.benchmark_data_path, args.dataset)
    kb_path = os.path.join(benchmark_path, '%s_typed_kb.jsonl' %args.dataset)
    kb = {}
    with open(kb_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            js = json.loads(line)
            title = js['title']
            abstract = js['abstract']
            types = js['types']
            if (len(types) > 0):
                types = types[-1]
            else:
                types = None
            kb[title] = (abstract, types)

    type_id_map = {
        'Other': 0,
        'Person': 1,
        'Place': 2,
        'Work': 3,
        'Organization': 4,
        'Event': 5,
        'Fictional Character': 6,
        'Species': 7,
        'Activity': 8,
        'Device': 9,
        'Topical Concept': 10,
        'Ethnic Group': 11,
        'Food': 12,
        'Disease': 13,
    }

    all_data = []
    untyped_path = os.path.join(benchmark_path, '%s_untyped.jsonl' %args.dataset)
    with open(untyped_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            js = json.loads(line)
            mention = js['mention']
            left_context = js['left_context']
            right_context = js['right_context']
            answer = js['answer']
            candidates = js['entity_title']

            if (answer != 'NIL'):
                context_type_ids = [0]*len(type_id_map)
                context_type = kb.get(answer, ('', None))[1]
                if (context_type is not None) and (context_type in type_id_map):
                    context_type_ids[type_id_map[context_type]] = 1
                else:
                    context_type_ids[0] = 1
            else:
                context_type_ids = [0]*len(type_id_map)
                context_type = None

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
                if (types is not None) and (types in type_id_map):
                    type_ids[type_id_map[types]] = 1 # retain the top-level type
                else:
                    type_ids[0] = 1

                abstract = abstract[:args.abstract_limit]

                candidate_abstracts.append(abstract)
                labels.append(flag)
                candidate_types.append(types)
                candidate_type_ids.append(type_ids)

            left_context = left_context[-args.context_limit:]
            right_context = right_context[:args.context_limit]

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

    out_path = os.path.join(benchmark_path, '%s_typed.jsonl' %args.dataset)
    with open(out_path, 'w', encoding='utf-8') as fout:
        for data in all_data:
            fout.write(json.dumps(data, ensure_ascii=False))
            fout.write('\n')
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--benchmark_data_path', type=str, default='../data/processed_benchmark')
    parser.add_argument('--type_index_path', type=str, default='../data/processed_benchmark/type_index.jsonl')
    parser.add_argument('--dataset', type=str, choices=[
        'ace2004_questions',
        'AIDA-YAGO2_testa',
        'AIDA-YAGO2_testb',
        'AIDA-YAGO2_train',
        'AIDA-YAGO2_testa_nil',
        'AIDA-YAGO2_testb_nil',
        'AIDA-YAGO2_train_nil',
        'aquaint_questions',
        'clueweb_questions',
        'msnbc_questions',
        'wnedwiki_questions',
    ], required=True)
    
    parser.add_argument('--abstract_limit', type=int, default=2000)
    parser.add_argument('--context_limit', type=int, default=1000)

    args = parser.parse_args()
    main(args)