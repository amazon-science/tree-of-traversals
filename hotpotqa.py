import argparse
import random
import time
import json
import string
import re
import os

from langchain.callbacks import get_openai_callback

from treeoftraversals import TreeOfTraversals

parser = argparse.ArgumentParser()
parser.add_argument('--results_file', type=str, help='File to save results in', default='qald10_results.json')
parser.add_argument('--sample_breadth', type=int, help='Samples per node', default=1)
parser.add_argument('--max_depth', type=int, help='Max depth of actions before answering', default=5)  # 5 roughly corresponds to two hops

args = parser.parse_args()

idxs = list(range(7405))
random.Random(234).shuffle(idxs)

rs = []
infos = []
old_time = time.time()

USE_CACHE = True
RESULTS_FILE = args.results_file

data = open('data/hotpot_dev_fullwiki_v1.json', 'r').read()
data = json.loads(data)

results = {}
if os.path.isfile(RESULTS_FILE) and USE_CACHE:
    results = json.load(open(RESULTS_FILE, 'r'))


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", "", text)

    def white_space_fix(text):
        return re.sub("\s+", " ", text).strip()

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def score_answers(answer, label):
    a = normalize_answer(answer)
    l = normalize_answer(label)
    return 1 if (a == l) else 0


def get_question_from_hotpot(data, i):
    question = data[i]['question']
    answer = data[i]['answer']
    print(f"Question: {data[i]['question']}")
    print(f"Correct Answer: {data[i]['answer']}")
    return question, answer

for i in idxs[:20]:
    if str(i) in results and USE_CACHE:
        continue
    query, label = get_question_from_hotpot(data, i)
    with get_openai_callback() as cb:
        tree = TreeOfTraversals(sample_breadth=args.sample_breadth, max_depth=args.max_depth)
        final_answer = tree.run(query)
    print(final_answer)
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")
    score = score_answers(final_answer, label)
    state = tree.answer_state()
    results[i] = {
        'question': query,
        'answer': label,
        'predicted_answer': final_answer,
        'score': score,
        'kg': state.kg_str(),
        'trajectory': state.trajectory
    }
    json.dump(results, open(RESULTS_FILE, 'w'), indent=4)
    #
    #
    #
    # r, info = webthink(i, to_print=True)
    # rs.append(info['em'])
    # infos.append(info)
    # print(sum(rs), len(rs), sum(rs) / len(rs), (time.time() - old_time) / len(rs))
    # print('-----------')
    # print()