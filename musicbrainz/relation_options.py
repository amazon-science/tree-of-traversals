import json
import logging
import re
from time import sleep

import requests
from musicbrainzngs import MusicBrainzError
from retry import retry

ENTITY_CATEGORIES = [
    'area', 'artist', 'event', 'genre', 'instrument', 'label', 'place', 'recording', 'release', 'release-group',
    'series', 'url', 'work'
]

ENTITY_RELATION_CATEGORES = [f"{e}-rels" for e in ENTITY_CATEGORIES]

DIRECT_CATEGORIES = {
    'artist': ['recordings', 'releases', 'release-groups', 'works'],
    'recording': ['artists', 'releases', 'isrcs', 'url-rels'],
    'release': ['artists', 'collections', 'labels', 'recordings', 'release-groups'],
    'release-group': ['artists', 'releases']
}


@retry(MusicBrainzError, tries=5, backoff=2)
def get_data(entity_id, entity_type, includes="all"):
    sleep(0.5)
    base_url = "https://musicbrainz.org/ws/2/"
    relations_url = base_url + entity_type.replace('_', '-') + "/" + entity_id
    headers = {"User-Agent": "TreeOfTraversals"}
    if includes == "all":
        includes = "+".join(ENTITY_RELATION_CATEGORES)
        includes += "+genres" if entity_type != "url" else ""
        if entity_type in DIRECT_CATEGORIES:
            includes += "+"
            includes += "+".join(DIRECT_CATEGORIES[entity_type])
    elif includes is None:
        includes = None
    else:
        includes = includes
    # print(relations_url)
    # print(includes)
    params = {
        "fmt": "json",
        "inc": includes
    }
    response = requests.get(relations_url, headers=headers, params=params)
    if response.status_code != 200:
        if response.status_code == 503:
            logging.warning('Rate limit')
        raise MusicBrainzError(f"Error: status code {response.status_code}")
    data = response.json()
    return data


def get_relations(entity_id, entity_type):
    data = get_data(entity_id, entity_type)
    relation_types = set()
    for relation in data["relations"]:
        phrase = get_relation_phrase(relation['type-id'], relation['direction'])
        # phrase = display_relationship(relation, phrase) + f" ({relation['target-type']})"
        phrase = relation['type']
        # relation_types.add(('P0' + re.sub(r'\D', '', relation['type-id'])[:4], phrase))
        relation_types.add((relation['type-id'], phrase))

    # Print or process the relation types
    # for relation_type in relation_types:
    #     print(relation_type)

    props = [p[0] for p in relation_types]
    labels = [p[1] for p in relation_types]
    return props, labels

RELATION_PHRASE_FILE = 'musicbrainz/relation_phrase_cache.json'
relation_cache = json.load(open(RELATION_PHRASE_FILE))

def get_relation_phrase(relation_id, direction):
    if relation_id in relation_cache:
        return relation_cache[relation_id][direction]

    relations_endpoint = 'https://musicbrainz.org/relationship/'
    url = relations_endpoint + relation_id
    rdata = requests.get(url)
    content = str(rdata.content)
    pattern1 = re.escape('Forward link phrase:</strong> <!-- -->') + '(.*?)' + re.escape('</li>')
    pattern2 = re.escape('Reverse link phrase:</strong> <!-- -->') + '(.*?)' + re.escape('</li>')
    pattern3 = re.escape('Long link phrase:</strong> <!-- -->') + '(.*?)' + re.escape('</li>')
    forward = re.findall(pattern1, content)[0]
    backward = re.findall(pattern2, content)[0]
    long = re.findall(pattern3, content)[0]
    relation_cache[relation_id] = {'forward': forward, 'backward': backward, 'long': long}
    json.dump(relation_cache, open(RELATION_PHRASE_FILE, 'w'))
    return relation_cache[relation_id][direction]


def display_relationship(relationship_data, link_phrase):
    # Use regular expression to find placeholders enclosed in curly braces
    link_phrase = link_phrase.replace(':%|', '}{')
    placeholders = re.findall(r'{(.*?)}', link_phrase)
    # Create a dictionary to map placeholders to their corresponding values in the relationship data
    placeholder_values = {}
    # for placeholder in placeholders:
    #     if placeholder in relationship_data['attributes']:
    #         placeholder_values[placeholder] = relationship_data['attributes'][placeholder]
    #     else:
    #         placeholder_values[placeholder] = ''

    # Format the relationship using the placeholder_values
    # try:
    #     formatted_relationship = link_phrase.format(**placeholder_values)
    # except KeyError:

    # return formatted_relationship
    return relationship_data['type']


if __name__ == "__main__":
    get_relations("cc2c9c3c-b7bc-4b8b-84d8-4fbd8779e493", 'artist')
    # get_relations('45e258a3-4552-442c-ba22-7e5a6899c3e8', 'work')