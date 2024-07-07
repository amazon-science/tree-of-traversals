import uuid
from SPARQLWrapper import SPARQLWrapper, SPARQLExceptions, JSON

agent_id = uuid.uuid4()

sparql = SPARQLWrapper("https://query.wikidata.org/sparql", agent=f"tree-of-traversal sparql agent: {agent_id}")
sparql.setReturnFormat(JSON)

# from wikidata.sparql_wiki import sparql

mb_wikidata_props = [
    "P434",
    "P435",
    "P436",
    "P966",
    "P982",
    "P1004",
    "P1330",
    "P1407",
    "P4404",
    "P5813",
    "P6423",
    "P8052"
]

wd2mb_query = """SELECT ?ent ?mbidprops ?uri
WHERE {{
  VALUES ?mbidprops {{ p:P434 p:P435 p:P436 p:P966 p:P982 p:P1004 p:P1330 p:P1407 p:P4404 p:P5813 p:P6423 p:P8052 }}
  VALUES ?mbiduri {{ psn:P434 psn:P435 psn:P436 psn:P966 psn:P982 psn:P1004 psn:P1330 psn:P1407 psn:P4404 psn:P5813 psn:P6423 psn:P8052 }}
  VALUES ?ent {{ {entities} }}
  ?ent ?mbidprops ?mbidstatement.
  ?mbidstatement ?mbiduri ?uri
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
}}
"""

def link_wikidata_to_musicbrainz(entities):
    ent_str = " ".join([f"wd:{e}" for e in entities])
    query = wd2mb_query.format(entities=ent_str)
    sparql.setQuery(query)
    results = sparql.query().convert()
    results = results['results']['bindings']
    wd2mb = {}
    for r in results:
        ent = r['ent']['value'].split('/')[-1]
        mbtype, id = r['uri']['value'].split('/')[-2:]
        wd2mb[ent] = {'musicbrainz': {'id': id, 'mbtype': mbtype}}
    return wd2mb


if __name__ == "__main__":
    entities = ['Q30', 'Q23215', 'Q19520501']
    res = link_entities(entities)

    breakpoint()