import os
import argparse

from langchain.callbacks import get_openai_callback

import llm
from treeoftraversals import TreeOfTraversals

parser = argparse.ArgumentParser()
parser.add_argument('--query', type=str, help='Query string')
parser.add_argument('--sample_breadth', type=int, help='Samples per node', default=1)
parser.add_argument('--max_depth', type=int, help='Max depth', default=7)
parser.add_argument('--llm', type=str, help='model id to use (openai/huggingface)', default='gpt-3.5-turbo')
parser.add_argument('--knowledge_bases', type=str, nargs='+', help="Knowledge base or bases to use. Currently supports 'wikidata' and 'musicbrainz' (default 'wikidata')" , default='wikidata')

def main(args):
    query = args.query
    if not query:
        query = 'Which actor played in both Inception and Interstellar?'
    with get_openai_callback() as cb:
        model = llm.get_llm(args.llm, n=args.sample_breadth)
        tree = TreeOfTraversals(llm=model, sample_breadth=args.sample_breadth, max_depth=args.max_depth, knowledge_bases=args.knowledge_bases)
        final_answer = tree.run(query)
    print(final_answer)
    return final_answer, tree

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
