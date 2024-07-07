from SPARQLWrapper import SPARQLExceptions
from retry import retry
from wikidata.sparql_wiki import sparql

# Gets predicates of subject
query_forward_properties = """
SELECT ?property ?predicateType ?label WHERE {{
  {{
    SELECT DISTINCT ?property ?predicateType WHERE {{
      VALUES ?subject {{ {entities} }}
      ?subject ?predicateType ?object.
      ?property wikibase:directClaim ?predicateType.
    }}
  }}
  SERVICE wikibase:label {{
    bd:serviceParam wikibase:language "en".
    ?property rdfs:label ?label.
  }}

  FILTER (!regex(str(?label), "ID"))
}}
ORDER BY ASC(xsd:integer(STRAFTER(STR(?property), 'http://www.wikidata.org/entity/P')))
LIMIT 100
"""

# Gets predicates of object
query_backward_properties = """
SELECT ?property ?predicateType ?label WHERE {{
  {{
    SELECT DISTINCT ?property ?predicateType WHERE {{
      VALUES ?object {{ {entities} }}
      ?subject ?predicateType ?object.
      ?property wikibase:directClaim ?predicateType.
    }} LIMIT 50
  }}
  SERVICE wikibase:label {{
    bd:serviceParam wikibase:language "en".
    ?property rdfs:label ?label.
  }}

  FILTER (!regex(str(?label), "ID"))
}}
ORDER BY ASC(xsd:integer(STRAFTER(STR(?property), 'http://www.wikidata.org/entity/P')))
LIMIT 100
"""

def get_forward_relations(entities):
    return get_available_relations(entities, query_forward_properties)


def get_backward_relations(entities):
    return get_available_relations(entities, query_backward_properties)


@retry(exceptions=SPARQLExceptions.EndPointInternalError, tries=5, delay=1)  # TODO: add logging on failure
def get_available_relations(entities, base_query):
    entities_str = " ".join(f"wd:{e}" for e in entities)
    query = base_query.format(entities=entities_str)
    sparql.setQuery(query)
    results = sparql.query().convert()
    properties = [r['property']['value'].split('/')[-1] for r in results['results']['bindings']]
    labels = [r['label']['value'] for r in results['results']['bindings']]
    return properties, labels


if __name__ == '__main__':
    entities = ['Q25188', 'Q13417189']
    props, labels = get_backward_relations(entities)
    options = '\n'.join([f"{p}: {l}" for p, l in zip(props, labels)])
    print(options)