import json
import os
from collections import Counter


if __name__ == '__main__':
    kb_path = '/data/zfw/NEL/data/processed/kb.jsonl'
    kb = {}
    with open(kb_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            js = json.loads(line)
            title, types = js['title'], js['type']
            kb[title] = types

    dataset_path = '/data/zfw/NEL/data/processed'
    dataset_splits = ['train', 'valid', 'test']
    type_freq = Counter()

    for spl in dataset_splits:
        full_path = os.path.join(dataset_path, '%s.jsonl' %spl)
        with open(full_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                js = json.loads(line)
                for title in js['candidate_title']:
                    types = kb.get(title, [])
                    # for t in types:
                    #     type_freq[t] += 1
                    if (len(types) > 0):
                        type_freq[types[-1]] += 1

    sorted_freq = sorted(type_freq.items(), key=lambda x:x[1], reverse=True)
    total_count = sum([x[1] for x in sorted_freq])
    top = sorted_freq[:10]
    for typ, freq in top:
        print('%s: %d/%d = %f' %(typ, freq, total_count, freq/total_count))