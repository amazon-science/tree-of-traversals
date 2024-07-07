import os
import argparse

from langchain.callbacks import get_openai_callback

import llm
from treeoftraversals import TreeOfTraversals

# import streamlit as st
#
# st.title("Tree-of-traversals")
#
# text = st.text_input("Enter query:")


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
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")
    return final_answer, tree

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

# if text:
#     answer, tree = main(query=text)
#     for node in tree.state_tree.traverse():
#         st.write(node.state.trajectory)
