import json


def is_correct(js):
    return js['answer'] == js['prediction'], js['answer'] == 'NIL'


def main():
    result_paths = {
        'blink': '/data/zfw/NEL/data/prediction/blink/NEL_prediction.txt',
        'clink': '/data/zfw/NEL/data/prediction/clink/NEL_prediction.txt',
        'cross_blink': '/data/zfw/NEL/data/prediction/cross_blink/NEL_prediction.txt',
        'cross_clink': '/data/zfw/NEL/data/prediction/cross_clink/NEL_prediction.txt',
    }

    results = {}
    for key in result_paths:
        results[key] = []
        path = result_paths[key]
        with open(path, 'r', encoding='utf-8') as fin:
            for line in fin:
                js = json.loads(line)
                results[key].append(js)

    lengths = [len(results[x]) for x in results]
    print(lengths)
    assert lengths[0] == lengths[1]
    assert lengths[0] == lengths[2]
    assert lengths[0] == lengths[3]

    pick1 = []
    pick2 = []
    for i in range(lengths[0]):
        c_correct, c_nil = is_correct(results['clink'][i])
        b_correct, b_nil = is_correct(results['blink'][i])
        cc_correct, cc_nil = is_correct(results['cross_clink'][i])
        bc_correct, bc_nil = is_correct(results['cross_blink'][i])

        pp = {
            'clink': results['clink'][i]['prediction'],
            'blink': results['blink'][i]['prediction'],
            'cross_clink': results['cross_clink'][i]['prediction'],
            'cross_blink': results['cross_blink'][i]['prediction'],
        }
        gathered_json = {
            'left_context': results['blink'][i]['left_context'],
            'right_context': results['blink'][i]['right_context'],
            'mention': results['blink'][i]['mention'],
            'answer': results['blink'][i]['answer'],
            'prediction': pp,
        }

        if (cc_correct) and not(bc_correct):
            pick1.append(gathered_json)

        # if (cc_correct) and not(c_correct):
        #     pick2.append(gathered_json)
        if (cc_correct) and not(c_correct) and cc_nil:
            pick2.append(gathered_json)


    # print('Examples that type matters: (%d in total)' %len(pick1))
    # for js in pick1:
    #     print(json.dumps(js, indent=2))
    #     input()

    print('Examples that cross_encoder matters: (%d in total)' %len(pick2))
    for js in pick2:
        print(json.dumps(js, indent=2))
        input()
        

if __name__ == '__main__':
    main()