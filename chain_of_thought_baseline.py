from langchain.schema import HumanMessage

import argparse

from llm import get_llm

parser = argparse.ArgumentParser()
parser.add_argument('--query', type=str, help='Query string')

def chain_of_thought(query, llm):
    prompt = "Query: " + query + "\nLet's think step by step:"
    result1 = llm(prompt, n=1)[0]
    prompt2 = prompt + result1 + "\nSo the answer is (just the answer, separate multiple answers with '|'):"
    result = llm(prompt2, n=1, max_tokens=128)[0]
    return result, result1


if __name__=="__main__":
    llm = get_llm('gpt-3.5-turbo')
    args = parser.parse_args()
    print(chain_of_thought(args.query, llm))