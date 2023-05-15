# coding: utf-8

import os
import pickle
from tqdm import tqdm
import argparse
import json
from collections import defaultdict


def title2url(title):
    title = title.replace(' ', '_')
    url = 'https://en.wikipedia.org/wiki/%s' %title
    return url


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

    url2type = {}
    type_index = {}
    with open(args.type_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            segs = line.strip().split('\t\t')
            url, typeline = segs[0], segs[1]
            url = url[1:-1]
            url2type[url] = typeline
            types = typeline.split('->')
            for t in types:
                if (t not in type_index) and (t != 'Thing'):
                    type_index[t] = len(type_index)

    if not(os.path.exists(args.type_index_path)):
        with open(args.type_index_path, 'w', encoding='utf-8') as fout:
            for t in type_index:
                js = {
                    'type': t,
                    'index': type_index[t],
                }
                json.dump(js, fout, ensure_ascii=False)
                fout.write('\n')

    untyped_entities = []
    kb_path = os.path.join(args.kb_path, args.dataset, '%s_kb.txt' %args.dataset)
    out_path = os.path.join(args.kb_path, args.dataset, '%s_typed_kb.jsonl' %args.dataset)
    with open(out_path, 'w', encoding='utf-8') as fout:
        with open(kb_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                js = json.loads(line)
                abstract, title = js['abstract'], js['title']
                url = title2url(title)
                typeline = url2type.get(url, '')
                types = typeline.split('->')
                if (len(types) == 0):
                    untyped_entities.append(title)
                else:
                    types = types[:-1]

                new_js = {
                    'title': title,
                    'abstract': abstract,
                    'types': types,
                }
                json.dump(new_js, fout, ensure_ascii=False)
                fout.write('\n')

    untyped_path = os.path.join(args.kb_path, args.dataset, '%s_untyped_entities.txt' %args.dataset)
    with open(untyped_path, 'w', encoding='utf-8') as fout:
        for t in untyped_entities:
            fout.write('%s\n' %t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kb_path', type=str, default='../data/processed_benchmark')
    parser.add_argument('--type_path', type=str, default='../data/raw/ins-cls.txt')
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

    args = parser.parse_args()
    main(args)