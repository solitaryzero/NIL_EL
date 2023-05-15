import json
import os
import re


def main():
    result_paths = {
        'CLINK_bi': '/data/zfw/NEL/data/evaluation/clink/NEL_grid_results.txt',
        'CLINK_cross': '/data/zfw/NEL/data/evaluation/cross_clink_mask/NEL_grid_results.txt',
    }

    all_results = {}
    lambda_pattern = re.compile(r'Result at lambda=(\d\.\d+):')
    value_pattern = re.compile(r'"([^"]+)": (\d\.\d*),')
    
    for key in result_paths:
        path = result_paths[key]
        all_results[key] = {}
        lam = None
        with open(path, 'r', encoding='utf-8') as fin:
            for line in fin:
                line = line.strip()
                res = re.match(lambda_pattern, line)
                if (res is not None):
                    lam = float(res.group(1))
                    all_results[key][lam] = {}
                
                res = re.match(value_pattern, line)
                if (res is not None):
                    _key = res.group(1)
                    _value = float(res.group(2))
                    all_results[key][lam][_key] = _value

    out_path = '/data/zfw/NEL/data/evaluation/merged_results'
    if not(os.path.exists(out_path)):
        os.makedirs(out_path)

    for key in all_results:
        out_file = os.path.join(out_path, '%s_all_results.txt' %(key))
        with open(out_file, 'w', encoding='utf-8') as fout:
            json.dump(all_results[key], fout)

if __name__ == '__main__':
    main()