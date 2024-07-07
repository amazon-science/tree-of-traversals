import uuid
from SPARQLWrapper import SPARQLWrapper, SPARQLExceptions, JSON

agent_id = uuid.uuid4()

sparql = SPARQLWrapper("https://query.wikidata.org/sparql", agent=f"tree-of-traversal sparql agent: {agent_id}")
sparql.setReturnFormat(JSON)