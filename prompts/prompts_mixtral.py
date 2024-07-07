ACTION_DESCRIPTIONS_TEXT = """You are a superintelligent AI equipped with the ability to search a knowledge graph for definitive, \
up-to-date answers. Your task is to interface with the knowledge graph in order to answer the above query. \
You will be able to expand the knowledge graph until you have found the answer. Think in detail before acting or answering.\

Available actions:
'THINK' - Generate relevant thoughts to solving the problem. This could include recalling well known facts from memory.
\te.g.
\t\tTHINK: I should search for the movies directed by...
\t\tTHINK: I know that Biden is American, therefore...
\t\tTHINK: I see that John Cena is in both The Independent and Vacation Friends, therefore...
'EXPAND' - Search for edges of entities in the knowledge graph using an external API. This is a useful function for getting to the correct answer.
\te.g. EXPAND: I should search for the country of origin of Jonathon Taylor
'ANSWER' - Generate the final answer once the problem is solved. Just state the best answer, do not output a full sentence.
\te.g. 
\t\tANSWER: No
\t\tANSWER: Harry Potter
\t\tANSWER: [Harry Potter, Ron Weasley, Hermione Granger]

"""


BASE_ACTION_SELECTION_PROMPT = ACTION_DESCRIPTIONS_TEXT + """Next action [{options}]:"""

ANSWER_PROMPT = """Give your best answer based on the knowledge graph. Give your reasoning and then state the best answer.

e.g.
THOUGHT: ...
ANSWER: ...

THOUGHT:"""

THINK_PROMPT = """THOUGHT:"""

EXPAND_KG_ENTITY_SELECTION_PROMPT = """Current task
EXPAND takes two parameters. The first is the entity or entity group to get more information about. Select which entity or entities from the KG to expand, you can select more than one at a time.
Provide the QIDs. Options include {options}
SELECT ENTITIES:"""

EXPAND_KG_RELATION_SELECTION_PROMPT = """Your current task is to select the property (PID) to expand along for the selected entities.
The selected entities are: {selected_entities}

The options of properties to choose from are:
{outgoing}
{incoming}

Select exactly one property (PID e.g. P10) from those listed above
SELECT PROPERTY:"""

EVALUATE_STATE_PROMPT = """Your current task is to evaluate the above knowledge graph and action history.
Based on the original query, the current knowledge graph, and the action history, \
give the likelihood that the model will correctly answer the question. 
If the most recent action provided information towards the goal and followed the preceding thought, give a high score. 
If the last action was unhelpful, give a low score. 

The output should be a number between 0.1 and 1 with one decimal. Do not output anything else. 

RATING [0.1-1.0]:"""

EVALUATE_ANSWER_PROMPT = """
Query: {query}

{kg_state}

Provided answer: {answer}

Your task is to score the correctness of the provided answer based on the original query, and the knowledge graph.
Give a pessimistic score from 0.0 to 1.0 on how likely the answer is to be correct. 
0.0 if definitely wrong
0.0 if unable to answer based on the knowledge graph
0.5 if unsure
0.7 for probably correct but not confirmed in knowledge graph
1.0 for definitely correct and confirmed in knowledge graph.

Give reasoning to get to the correct answer. Then provide a score.
E.g. 
Reasoning...
So the score for the provided answer should be...
"""

FULL_PROMPT = """Original Query: {query}

{kg_state}

Original Query: {query}

Previous actions:
{trajectory}

{current_prompt}"""

