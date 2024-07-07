from wikidata.sparql_wiki import sparql

label_description_query = """\
SELECT ?qid ?label ?description WHERE {{
  VALUES ?qid {{ {} }}  # Replace with your list of QIDs
  
  ?qid rdfs:label ?label.
  FILTER(LANG(?label) = "en").
  
  OPTIONAL {{
    ?qid schema:description ?description.
    FILTER(LANG(?description) = "en").
  }}
}}
"""


def extract_qid(url):
    return url.split('/')[-1]


def query_label_and_descriptions(qids):
    """

    :param qids: a list of wikidata qids
    :return: Dictionary of the form
    {
      <qid>: {'label': <label>, 'description', <description>}
      ...
    }
    """
    wdqids = [f'wd:{qid}' for qid in qids]
    query = label_description_query.format(" ".join(wdqids))
    sparql.setQuery(query)
    results = sparql.query().convert()

    processed_results = {}
    for result in results["results"]["bindings"]:
        qid = extract_qid(result["qid"]["value"])
        label = result["label"]["value"] if "label" in result else ""
        description = result["description"]["value"] if "description" in result else ""
        processed_results[qid] = {'label': label, 'description': description}
        # combined_description = f"{label} - {description}"
        # processed_results[qid] = {"label": label, "description": description, "combined_description": combined_description}
    return processed_results
