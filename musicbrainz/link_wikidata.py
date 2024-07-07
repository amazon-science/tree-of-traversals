import requests
from musicbrainzngs import MusicBrainzError

from musicbrainz.relation_options import get_data


def link_musicbrainz_to_wikidata(entity_id, entity_type):
    data = get_data(entity_id, entity_type, includes="url-rels")
    if 'relations' not in data:
        return None
    rels = data['relations']
    wikidata_rels = [r for r in rels if r['type'] == 'wikidata']
    if not wikidata_rels:
        return None
    wikidata_qid = wikidata_rels[0]['url']['resource'].split('/')[-1]
    return wikidata_qid