import os
import json
import csv
from tqdm import tqdm

import random

if __name__ == '__main__':
    data_path = './data/processed_benchmark'
    splits = ['train', 'testb']

    kb = {}
    for spl in splits:
        kb_path = os.path.join(data_path, 'AIDA-YAGO2_%s_nil' %spl, 'AIDA-YAGO2_%s_nil_kb.txt' %spl)
        with open(kb_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                js = json.loads(line)
                abst = js['abstract'].split('::;')[0]
                kb[js['title']] = abst

    # prefix = 'https://www.xlore.cn/instance.html?url=http://xlore.org/instance/'
    # url_path = './data/raw/en_instance_ID.txt'
    # url_map = {}
    # with open(url_path, 'r', encoding='utf-8') as fin:
    #     for line in fin:
    #         segs = line.split('\t\t')
    #         title, eid = segs[0], segs[3]
    #         url = prefix+eid
    #         url_map[title] = url

    nil_data = []
    for spl in splits:
        full_path = os.path.join(data_path, 'AIDA-YAGO2_%s_nil' %spl, 'AIDA-YAGO2_%s_nil_untyped.jsonl' %spl)
        with open(full_path, 'r', encoding='utf-8') as fin:
            for i, line in tqdm(enumerate(fin)):
                js = json.loads(line)
                answer = js['answer']
                if (answer == 'NIL'):
                    js['id'] = i
                    js['split'] = spl
                    nil_data.append(js)

    nil_data = random.sample(nil_data, 300)

    csv_data = []
    header = ['id', 'source', 'left_context', 'mention', 'right_context', 'nil type', 'candidate title', 'abstract']
    csv_data.append(header)

    for js in nil_data:
        row = [
            js['id'],
            js['split'],
            js['left_context'],
            js['mention'],
            js['right_context'],
            ''
        ]
        for title in js['entity_title']:
            abst = kb.get(title, '[UNKNOWN]')
            row.append(title)
            row.append(abst)

        csv_data.append(row)
    
    out_path = './data/AIDA_annotation/to_annotate.csv'
    with open(out_path, 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerows(csv_data)