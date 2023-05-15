import json
import os
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import sys

BEGIN_ENT_TOKEN = "[START_ENT]"
END_ENT_TOKEN = "[END_ENT]"


def read_extracted_questions(filename):
    extracted_data = {}
    with open(filename, 'r', encoding='utf-8') as fin:
        for line in fin:
            js = json.loads(line)
            qid = js['query_id']
            extracted_data[qid] = js

    return extracted_data


def extract_questions(filename, extracted_non_nil_data):

    # all the datapoints
    global_questions = []

    # left context so far in the document
    left_context = []

    # working datapoints for the document
    document_questions = []

    # is the entity open
    open_entity = False

    # question id in the document
    question_i = 0
    question_nil_i = 0

    with open(filename) as fin:
        lines = fin.readlines()

        for line in tqdm(lines):

            if "-DOCSTART-" in line:
                # new document is starting

                doc_id = line.split("(")[-1][:-2]

                # END DOCUMENT

                # check end of entity
                if open_entity:
                    document_questions[-1]["input"].append(END_ENT_TOKEN)
                    open_entity = False

                """
                #DEBUG
                for q in document_questions:
                    pp.pprint(q)
                    input("...")
                """

                # add sentence_questions to global_questions
                global_questions.extend(document_questions)

                # reset
                left_context = []
                document_questions = []
                question_i = 0
                question_nil_i = 0

            else:
                split = line.split("\t")
                token = split[0].strip()

                if (len(split) >= 5) or (split[-1].strip() == '--NME--'):
                    is_nil = (split[-1].strip() == '--NME--')
                    B_I = split[1]
                    mention = split[2]
                    if not(is_nil):
                        #  YAGO2_entity = split[3]
                        Wikipedia_URL = split[4]
                        Wikipedia_ID = split[5]
                        # Freee_base_id = split[6]

                    if (B_I == "I"):
                        pass

                    elif (B_I == "B"):
                        if not(is_nil):
                            title = Wikipedia_URL.split("/")[-1].replace("_", " ")
                            query_id = "{}:{}".format(doc_id, question_i)
                            q = extracted_non_nil_data[query_id]
                            q['id'] = query_id
                            q['left_context'] = left_context.copy()
                            q['right_context'] = []
                            q['input'] = left_context.copy() + [BEGIN_ENT_TOKEN]
                            document_questions.append(q)
                            open_entity = True
                            question_i += 1
                        else:
                            query_id = "{}:{}:nil".format(doc_id, question_i)
                            q = {
                                "id": query_id,
                                "input": left_context.copy() + [BEGIN_ENT_TOKEN],
                                "mention": mention,
                                "Wikipedia_title": 'NIL',
                                "Wikipedia_URL": None,
                                "Wikipedia_ID": None,
                                "left_context": left_context.copy(),
                                "right_context": [],
                            }
                            document_questions.append(q)
                            open_entity = True
                            question_nil_i += 1

                    else:
                        print("Invalid B_I {}", format(B_I))
                        sys.exit(-1)

                    # print(token,B_I,mention,Wikipedia_URL,Wikipedia_ID)
                else:
                    if open_entity:
                        document_questions[-1]["input"].append(END_ENT_TOKEN)
                        open_entity = False

                left_context.append(token)
                for q in document_questions:
                    q["input"].append(token)

                for q in document_questions[:-1]:
                    q["right_context"].append(token)

                if len(document_questions) > 0 and not open_entity:
                    document_questions[-1]["right_context"].append(token)

    # FINAL SENTENCE
    if open_entity:
        open_entity = False

    # add sentence_questions to global_questions
    global_questions.extend(document_questions)

    return global_questions


# store on file
def store_questions(questions, OUT_FILENAME):

    if not os.path.exists(os.path.dirname(OUT_FILENAME)):
        os.makedirs(os.path.dirname(OUT_FILENAME))

    with open(OUT_FILENAME, "w+") as fout:
        for q in questions:
            json.dump(q, fout)
            fout.write("\n")


def convert_to_BLINK_format(questions):
    data = []
    for q in questions:
        datapoint = {
            "context_left": " ".join(q["left_context"]).strip(),
            "mention": q["mention"],
            "context_right": " ".join(q["right_context"]).strip(),
            "query_id": q["id"],
            "label_id": q["Wikipedia_ID"],
            "Wikipedia_ID": q["Wikipedia_ID"],
            "Wikipedia_URL": q["Wikipedia_URL"],
            "Wikipedia_title": q["Wikipedia_title"],
        }
        data.append(datapoint)
    return data


# AIDA-YAGO2
print("AIDA-YAGO2")
extracted_file_paths = [
    '../data/benchmark/AIDA/AIDA-YAGO2_train.jsonl',
    '../data/benchmark/AIDA/AIDA-YAGO2_testa.jsonl',
    '../data/benchmark/AIDA/AIDA-YAGO2_testb.jsonl',
]
all_extracted_data = {}
for p in extracted_file_paths:
    extracted_data = read_extracted_questions(p)
    for key in extracted_data:
        all_extracted_data[key] = extracted_data[key]

in_aida_filename = (
    "../data/train_and_benchmark_data/basic_data/test_datasets/AIDA/AIDA-YAGO2-dataset.tsv"
)
aida_questions = extract_questions(in_aida_filename, all_extracted_data)

train = []
testa = []
testb = []
for element in aida_questions:
    if "testa" in element["id"]:
        testa.append(element)
    elif "testb" in element["id"]:
        testb.append(element)
    else:
        train.append(element)
print("train: {}".format(len(train)))
print("testa: {}".format(len(testa)))
print("testb: {}".format(len(testb)))

train_blink = convert_to_BLINK_format(train)
testa_blink = convert_to_BLINK_format(testa)
testb_blink = convert_to_BLINK_format(testb)

out_train_aida_filename = "../data/benchmark/AIDA_nil/AIDA-YAGO2_train.jsonl"
store_questions(train_blink, out_train_aida_filename)
out_testa_aida_filename = "../data/benchmark/AIDA_nil/AIDA-YAGO2_testa.jsonl"
store_questions(testa_blink, out_testa_aida_filename)
out_testb_aida_filename = "../data/benchmark/AIDA_nil/AIDA-YAGO2_testb.jsonl"
store_questions(testb_blink, out_testb_aida_filename)
