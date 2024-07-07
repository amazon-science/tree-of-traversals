import re

import requests

from musicbrainz.relation_options import ENTITY_RELATION_CATEGORES, DIRECT_CATEGORIES, get_relation_phrase, get_data, display_relationship
from musicbrainz.initial_state import entity2string, mb_ent_to_qent, mb_prop_to_pprop


def get_edges(entity_id, entity_type, relation_id):
    data = get_data(entity_id, entity_type)
    rels = data['relations']
    rels = [d for d in rels if d['type-id'] == relation_id]
    if not rels:
        # if no relation found, return nothing.
        return [], [], [], [], {}

    target_ids = []
    target_strings = []
    attribute_strings = []
    new_entities = {}
    objects = []

    for rel in rels:
        target_type = rel['target-type']
        rel_direction = rel['direction']
        target = rel[target_type]
        target_string = entity2string(target, target_type)
        target_id = target['id']
        target_strings.append(target_string)
        target_ids.append(target_id)
        relation_phrase = get_relation_phrase(relation_id, rel_direction)
        attributes = rel['attributes']
        # relation_phrase = display_relationship(rel, relation_phrase)
        if re.findall(r'\{(.*)\}', relation_phrase):
            relation_phrase = rel['type']
            attributes = [rel_direction] + attributes
        if attributes:
            attribute_string = " (" + ", ".join(attributes) + ")"
        else:
            attribute_string = ""
        attribute_strings.append(attribute_string)
        try:
            label, desc = target_string.split(' - ')
        except ValueError:
            label = target_string
            desc = ""
        except AttributeError:
            breakpoint()
        qvalue = mb_ent_to_qent(target_id)  # Q0 is distinct from all wikidata entities
        new_entities[qvalue] = {
            'label': label,
            'description': desc,
            'musicbrainz': {
                'id': target_id,
                'mbtype': target_type
            }
        }
        objects.append(qvalue)

    subjects = [mb_ent_to_qent(entity_id)] * len(objects)
    relations = [mb_prop_to_pprop(relation_id)] * len(objects)

    return subjects, relations, objects, attribute_strings, new_entities



if __name__ == "__main__":
    from pprint import pprint as pp
    eras_tour_venue = get_edges("6d825765-2ebe-4266-a5b5-4293b711c97f", 'event', 'e2c6f697-07dc-38b1-be0b-83d740165532')  # eras tour: Los angeles (night 1), held at
    tswift_support = get_edges('20244d07-534f-4eff-b4d4-930878889970', 'artist', 'ed6a7891-ce70-4e08-9839-1f2f62270497')
    breakpoint()