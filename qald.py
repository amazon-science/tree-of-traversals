import pickle
import random
import time
import json
import string
import re
import os
import argparse
from datetime import datetime

import inflect
from langchain.callbacks import get_openai_callback

from chain_of_thought_baseline import chain_of_thought
from llm import get_llm
from treeoftraversals import TreeOfTraversals

from SPARQLWrapper import SPARQLWrapper, JSON, SPARQLExceptions

parser = argparse.ArgumentParser()
parser.add_argument('--results_file', type=str, help='File to save results in', default='qald10_results.json')
parser.add_argument('--sample_breadth', type=int, help='Samples per node', default=1)
parser.add_argument('--max_depth', type=int, help='Max depth of actions before answering', default=5)  # 5 roughly corresponds to two hops
parser.add_argument('--model', type=str, help='What model to use [chain-of-thought/tree-of-traversals]', default='tree-of-traversals')
parser.add_argument('--llm', type=str, help='What base llm to use', default='gpt-3.5-turbo')

args = parser.parse_args()



# sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
# sparql.setReturnFormat(JSON)
from wikidata.sparql_wiki import sparql

qid2label = """SELECT ?label
WHERE {{
  VALUES ?entity {{ {entities} }}
  ?entity rdfs:label ?label.
  FILTER(LANG(?label) = "en").
}}
"""

def get_labels_from_qids(qids):
    if not qids:
        return []
    entities = " ".join([f"wd:{qid}" for qid in qids if qid[0] == 'Q'])
    if len(entities) == 0:
        raise ValueError("Cannot handle lexemes, no QIDs found")
    sparql.setQuery(qid2label.format(entities=entities))
    results = sparql.query().convert()['results']['bindings']
    answers = [r['label']['value'] for r in results if 'label' in r]
    if not answers:
        answers = qids
    return answers

rs = []
infos = []
old_time = time.time()

USE_CACHE = True
RESULTS_FILE = args.results_file
RESULTS_DIR = os.path.join(*os.path.split(RESULTS_FILE)[:-1])
TREE_DIR = os.path.join(RESULTS_DIR, os.path.splitext(os.path.split(RESULTS_FILE)[-1])[0])

data = json.load(open('data/qald_10.json', 'r'))
data_len = len(data['questions'])

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
    def is_correct(a, l):
        if l in a:
            return True
        if re.match("\d+", l):
            p = inflect.engine()
            l_as_word = p.number_to_words(l)
            if l_as_word in a:
                return True
        return False

    if not isinstance(label, list):
        label = [label]
    label_set = {normalize_answer(l) for l in label}
    answer = normalize_answer(answer)
    correct_answers = [is_correct(answer, l) for l in label_set]
    num_correct = sum(correct_answers)
    num_gold_answers = len(label_set)
    return num_correct/num_gold_answers
    #
    #
    #     set1 = {normalize_answer(a) for a in answers}
    #     set2 = {normalize_answer(l) for l in label} if isinstance(label, list) else {normalize_answer(label)}
    #     score = len(set1.intersection(set2)) / len(set1.union(set2))
    #     return score
    # else:
    #     a = normalize_answer(answer)
    #     l = normalize_answer(label)
    #     return 1 if (l in a) else 0

def parse_date(date_time):


    def pad_year(sign, year):
        return sign + ('00000' + year)[-4:]
    date_time = re.sub(r'^(?![+-])', '+', date_time)
    date_time = re.sub(r'^([+-]?)(\d{1,3}\b)', lambda match: pad_year(match.group(1), match.group(2)), date_time)
    date_time = re.sub(r'Z$', '', date_time)
    # year = re.match(r'^[+-]?(\d+\b)', date_time)[0]

    return datetime.strptime(date_time[1:], '%Y-%m-%dT%H:%M:%S')


def format_date(date_time):

    is_bce = False

    def replace_year(match):
        nonlocal is_bce
        year = int(match.group(0))
        is_bce = True
        return str(abs(year))

    positive_date = re.sub(r'^(?:-\d+|\+?0+\b)', replace_year, date_time)
    try:
        moment = parse_date(positive_date)
    except ValueError:
        moment = None
    formatted = ""

    if moment:
        if moment.month == 1 and moment.day == 1:
            formatted = moment.strftime('%Y')
        else:
            formatted = moment.strftime('%B %d, %Y')
    else:
        year = re.sub(r'^\+?(\d+).*', r'\1', positive_date)
        formatted = '0' * (4 - len(year)) + year

    if is_bce:
        formatted += ' BCE'

    return formatted

def get_answer(answer):
    if 'boolean' in answer:
        return 'Yes' if answer['boolean'] else 'No'
    answer = answer['results']['bindings']
    final_answer = []
    get_labels = []
    for a in answer:
        if 'result' in a:
            t = a['result']['type']
            r = a['result']['value']
        else:
            t = list(a.values())[0]['type']
            r = list(a.values())[0]['value']
        if t == 'uri' and r.split('/')[-1] and r.split('/')[-1][0] in ['Q', 'L']:
            get_labels.append(r.split('/')[-1])
            continue  # get all qids and then get labels with single query
        elif t == 'uri':
            final_answer.append(r)
        elif t == "literal" and r[-1] == "Z":
            r = format_date(r)
            final_answer.append(r)
        elif t == "literal":
            final_answer.append(r)
    final_answer += get_labels_from_qids(get_labels)
    return final_answer

def updated_answer(sparql_q):
    sparql.setQuery(sparql_q)
    return sparql.query().convert()


def get_question_from_qald(data, i):
    question = data['questions'][i]['question'][0]['string']
    sparql_q = data['questions'][i]['query']['sparql']
    try:
        answers = updated_answer(sparql_q)
    except SPARQLExceptions.EndPointInternalError:
        raise ValueError("Unable to update answer")
    # answers = data['questions'][i]['answers'][0]
    answers = get_answer(answers)
    print(f"Question: {question}")
    print(f"Correct Answer: {answers}")
    return question, answers


llm = get_llm(args.llm, args.sample_breadth)
for i in idxs[:500]:
    if str(i) in results and USE_CACHE:
        # score = score_answers(results[str(i)]['predicted_answer'], label)
        r = results[str(i)]
        if isinstance(r['predicted_answer'], list):
            r['predicted_answer'], r['thoughts'] = r['predicted_answer']
            r['score'] = score_answers(r['predicted_answer'], r['answer'])
            json.dump(results, open(RESULTS_FILE, 'w'), indent=4)
        r['score'] = score_answers(r['predicted_answer'], r['answer'])
        continue
    try:
        query, label = get_question_from_qald(data, i)
    except ValueError:
        continue
    if not label:
        continue
    if args.model == 'tree-of-traversals':
        with get_openai_callback() as cb:
            llm.reset_counts()
            tree = TreeOfTraversals(llm=llm, sample_breadth=args.sample_breadth, max_depth=args.max_depth, knowledge_bases=['wikidata'])
            try:
                final_answer = tree.run(query)
            except Exception as e:
                print(e)
                continue
        score = score_answers(final_answer, label)
        state = tree.answer_state()
        if not os.path.isdir(TREE_DIR):
            os.mkdir(TREE_DIR)
        treefile = os.path.join(TREE_DIR, f"{i}_tree.pkl")
        pickle.dump(tree, open(treefile, 'wb'))
        results[i] = {
            'question': query,
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
            final_answer, thoughtprocess = chain_of_thought(query, llm)
        score = score_answers(final_answer, label)
        print(thoughtprocess)
        print(final_answer)
        results[i] = {
            'question': query,
            'answer': label,
            'predicted_answer': final_answer,
            'score': score,
            'thoughts': thoughtprocess
        }
    print(f'Expected: {label}')
    print(f'Got: {final_answer}')
    print(f"Score: {score}")
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")
    json.dump(results, open(RESULTS_FILE, 'w'), indent=4)
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
