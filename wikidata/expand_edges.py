from datetime import datetime

from SPARQLWrapper import SPARQLExceptions
from retry import retry

from wikidata.sparql_wiki import sparql

query_outgoing = """SELECT ?baseEntity ?statement ?object ?label ?description ?timeprecision
WHERE {{
  VALUES ?baseEntity {{ {entities} }}
  ?baseEntity p:{property} ?statement.
  ?statement ps:{property} ?object.
  ?baseEntity wdt:{property} ?object.
  OPTIONAL {{
    ?statement psv:{property}/wikibase:timePrecision ?timeprecision.
  }}
  OPTIONAL {{
    ?object rdfs:label ?label.
    FILTER(LANG(?label) = "en").
  }}
  OPTIONAL {{
    ?object schema:description ?description.
    FILTER(LANG(?description) = "en").
  }}
}}
LIMIT 100
"""

query_incoming = """
SELECT ?baseEntity ?statement ?object ?label ?description
WHERE {{
  VALUES ?baseEntity {{ {entities} }}
  ?statement ps:{property} ?baseEntity.
  ?object p:{property} ?statement.
  ?object wdt:{property} ?baseEntity.
  OPTIONAL {{
    ?object rdfs:label ?label.
    FILTER(LANG(?label) = "en").
  }}
  OPTIONAL {{
    ?object schema:description ?description.
    FILTER(LANG(?description) = "en").
  }}
}}
LIMIT 100
"""

# TODO: potentially include qualifiers on all statements
# SELECT ?entity ?entityLabel ?prop ?value ?valueLabel ?wdpqLabel ?qaulifierValueLabel
# WHERE
# {
#     VALUES ?entity
# {wd: Q23215 wd: Q6279}
# ?entity
# p: P26 ?statement.
# ?statement
# ps: P26 ?value.
#
# OPTIONAL
# {
# ?statement ?pq ?qaulifierValue.
# ?wdpq
# wikibase: qualifier ?pq.
# }
#
# SERVICE
# wikibase: label
# {bd: serviceParam wikibase: language "[AUTO_LANGUAGE],en".}
# }



@retry(SPARQLExceptions.EndPointInternalError, tries=3, backoff=2)
def get_edges(entities, property, use_incoming):
    """
    :param entities: entities from which to expand
    :param property: relation to expand along
    :param use_incoming: (bool) if true does queries of the form (?, r, o). If false does (s, r, ?)
    :return: list of edges, list of entity objects, entity labels, entity descriptions
    """
    query = query_incoming if use_incoming else query_outgoing
    entities_str = " ".join(f"wd:{e}" for e in entities)
    query = query.format(entities=entities_str, property=property)
    sparql.setQuery(query)
    results = sparql.query().convert()
    results = results['results']['bindings']
    sources = [r['baseEntity']['value'].split('/')[-1] for r in results]
    objects = [r['object']['value'].split('/')[-1] for r in results]
    relations = [property] * len(results)

    timeprecision = None
    try:
        timeprecision = [int(r['timeprecision']['value']) for r in results]
        objects = [handle_wikidata_datetime(time, prec) for time, prec in zip(objects, timeprecision)]
    except KeyError:
        timeprecision = None

    new_entities = {}
    new_entity_pos = 'object'
    for r in results:
        try:
            new_ent = r[new_entity_pos]['value'].split('/')[-1]  # just get the QID
            label = r['label']['value']
        except KeyError:  # if object is not an entity (e.g. datetime), it won't have label and description
            continue
        desc = r['description']['value'] if 'description' in r else ""
        new_entities[new_ent] = {'label': label, 'description': desc, 'wikidata': new_ent}
    attributes = [''] * len(results)  # eventually can add qualifiers here
    return sources, relations, objects, attributes, new_entities


def handle_wikidata_datetime(time, precision):
    if time[0] == '-':
        era = " BCE"
        time = time[1:]
    else:
        era = ""
    try:
        dt = datetime.strptime(time, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return time
    match precision:
        case 11:  # day precision (month day, year)
            time = dt.strftime("%B %d, %Y") + era
        case 10:  # month (month year)
            time = dt.strftime("%B %Y") + era
        case 9:  # year
            time = dt.strftime("%Y") + era
        case 8:  # decade
            time = dt.strftime("%Ys") + era
        case 7:  # century
            century = (dt.year - 1) // 100 + 1
            suffix = "th" if 11 <= century <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(century % 10, "th")
            time = f"{century}{suffix} century{era}"
    return time


if __name__ == '__main__':
    import pprint
    entities = ['Q25188', 'Q13417189']
    prop = 'P361'
    pprint.pprint(get_edges(entities, prop, True))
