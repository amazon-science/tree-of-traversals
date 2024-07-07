from wikidata.sparql_wiki import sparql

title_to_id_query = """SELECT ?lemma ?item WHERE {{
  VALUES ?lemma {{
    {titles}
  }}
  ?sitelink schema:about ?item;
    schema:isPartOf <https://en.wikipedia.org/>;
    schema:name ?lemma.
}}"""

def titles_to_qids(titles):
    titles_text = [title.replace('"', '\\"').replace("'","\\'") for title in titles]
    titles_text = " ".join([f"\'{title}\'@en" for title in titles_text])
    query = title_to_id_query.format(titles=titles_text)
    sparql.setQuery(query)
    results = sparql.query().convert()
    results = results['results']['bindings']
    title_to_qid_map = {}
    for r in results:
        title_to_qid_map[r['lemma']['value']] = r['item']['value'].split('/')[-1]
    return title_to_qid_map


if __name__ == "__main__":
    titles = ["Donald Trump", "Muza Niýazowa", "Tiger"]
    print(titles_to_qids(titles))