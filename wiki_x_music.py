import math
import pickle
import random
import time
import json
import string
import re
import os
import argparse
import pandas as pd

from langchain.callbacks import get_openai_callback

from chain_of_thought_baseline import chain_of_thought
from llm import get_llm
from treeoftraversals import TreeOfTraversals



parser = argparse.ArgumentParser()
parser.add_argument('--results_file', type=str, help='File to save results in', default='qald10_results.json')
parser.add_argument('--sample_breadth', type=int, help='Samples per node', default=1)
parser.add_argument('--max_depth', type=int, help='Max depth of actions before answering', default=5)  # 5 roughly corresponds to two hops
parser.add_argument('--model', type=str, help='What model to use [chain-of-thought/tree-of-traversals]', default='tree-of-traversals')
parser.add_argument('--llm', type=str, help='What base llm to use', default='gpt-3.5-turbo')

args = parser.parse_args()


rs = []
infos = []
old_time = time.time()

USE_CACHE = True
RESULTS_FILE = args.results_file
RESULTS_DIR = os.path.join(*os.path.split(RESULTS_FILE)[:-1])
TREE_DIR = os.path.join(RESULTS_DIR, os.path.splitext(os.path.split(RESULTS_FILE)[-1])[0])


data = pd.read_csv('data/MusicbrainzxWikidata.csv', header=0)
data_len = len(data)

idxs = list(range(data_len))
random.Random(233).shuffle(idxs)

results = {}
if os.path.isfile(RESULTS_FILE) and USE_CACHE:
    results = json.load(open(RESULTS_FILE, 'r'))


def normalize_answer(s):
    s = str(s)
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", "", text)

    def white_space_fix(text):
        return re.sub("\s+", " ", text).strip()

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.casefold())))


def score_answers(answer, label):
    # Intersection over Union if either answer is list
    if not isinstance(label, list):
        label = [label]
    label_set = {normalize_answer(l) for l in label}
    answer = normalize_answer(answer)
    correct_answers = [(l in answer) for l in label_set]
    num_correct = sum(correct_answers)
    num_gold_answers = len(label_set)
    return num_correct/num_gold_answers

    #     set1 = {normalize_answer(a) for a in answers}
    #     set2 = {normalize_answer(l) for l in label} if isinstance(label, list) else {normalize_answer(label)}
    #     score = len(set1.intersection(set2)) / len(set1.union(set2))
    #     return score
    # else:
    #     a = normalize_answer(answer)
    #     l = normalize_answer(label)
    #     return 1 if (l in a) else 0

def get_question_from_wiki_x_music(data, i):
    question = data['Question'][i]
    answer = data['Answer'][i]
    print(f"Question: {question}")
    print(f"Correct Answer: {answer}")
    return question, answer

def main():
    llm = get_llm(args.llm, args.sample_breadth)
    for i in idxs[:500]:
        try:
            query, label = get_question_from_wiki_x_music(data, i)
            if isinstance(query, float) and math.isnan(query):
                continue
        except ValueError:
            continue
        if str(i) in results and USE_CACHE:
            score = score_answers(results[str(i)]['predicted_answer'], label)
            continue
        elif args.model == 'tree-of-traversals':
            with get_openai_callback() as cb:
                llm.reset_counts()
                tree = TreeOfTraversals(llm=llm, sample_breadth=args.sample_breadth, max_depth=args.max_depth, knowledge_bases=['wikidata', 'musicbrainz'])
                try:
                    final_answer = tree.run(query)
                except Exception as e:
                    print(e)
                    continue
            score = score_answers(final_answer, label)
            state = tree.answer_state()
            if not os.path.isdir(RESULTS_DIR):
                os.mkdir(RESULTS_DIR)
            if not os.path.isdir(TREE_DIR):
                os.mkdir(TREE_DIR)
            treefile = os.path.join(TREE_DIR, f"{i}_tree.pkl")
            pickle.dump(tree, open(treefile, 'wb'))
            results[i] = {
                'question': query,
                'question_type': data['Type'][i],
                'answer': label,
                'predicted_answer': final_answer,
                'score': score,
                'kg': state.kg_str() if state is not None else "None",
                'trajectory': state.trajectory if state is not None else "None",
                'treefile': treefile,
                'prompt_tokens': llm.prompt_tokens,
                'completion_tokens': llm.completion_tokens
            }
        elif args.model == 'chain-of-thought':
            with get_openai_callback() as cb:
                final_answer, thoughts = chain_of_thought(query, llm)
            score = score_answers(final_answer, label)
            results[i] = {
                'question': query,
                'answer': label,
                'predicted_answer': final_answer,
                'score': score,
                'thoughts': thoughts
            }
        print(f'Expected: {label}')
        print(f'Got: {final_answer}')
        print(f"Score: {score}")
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")
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


if __name__ == "__main__":
    main()