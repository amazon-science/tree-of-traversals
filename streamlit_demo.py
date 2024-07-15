import streamlit as st

import llm
from treeoftraversals import TreeOfTraversals




def query_script(query, model, sample_breadth, max_depth, streamlit_dynamic_boxes):
    model = llm.get_llm("gpt-3.5-turbo", n=sample_breadth)
    tree = TreeOfTraversals(llm=model, sample_breadth=sample_breadth, max_depth=max_depth, knowledge_bases="wikidata", use_streamlit=streamlit_dynamic_boxes)
    final_answer = tree.run(query)
    return final_answer

def main():
    st.title("Tree-of-Traversals Demo")
    st.write("This demo uses Wikidata as the backend knowledge base.")
    st.write("Github repo here: https://github.com/amazon-science/tree-of-traversals")

    query = st.text_input("Enter your query:")
    sample_breadth = st.slider("Sample Breadth", 1, 3, 2)
    max_depth = st.slider("Max Depth", 6, 10, 7)
    model = "gpt-3.5-turbo"

    streamlit_dynamic_boxes = {
        'answer': st.empty(),
        'search_tree': st.empty(),
        'knowledge_graph': st.empty()
    }

    # search_tree = st.session_state.get('search_tree', "")
    # knowledge_graph = st.session_state.get('knowledge_graph', "")

    # search_tree = st.empty()
    # knowledge_graph = st.empty()
    # st.text_area("Search Tree", value=search_tree, height=200)

    if st.button("Run Query"):
        if query:
            streamlit_dynamic_boxes['answer'].text('Running...')
            result = query_script(query, model, sample_breadth, max_depth, streamlit_dynamic_boxes)
            streamlit_dynamic_boxes['answer'].text(f'Answer {result}')
        else:
            st.error("Please enter a query.")

if __name__ == "__main__":
    main()