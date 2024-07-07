import re

import musicbrainzngs as mbz

from musicbrainz.relation_options import get_data

mbz.set_useragent('TreeOfTraversals', '0.1')

# match_prompt_template = "MusicBrains entities:\n{options}\n\n In the context of this query: {context}\nChoose which of the above entities best matches the following\n{text}?\n\nOutput just the number."
match_prompt_template = "MusicBrains entities:\n{options}\n\n Choose which of the above entities best matches the following\n{text}?\n\nOutput just the number. Output 'None' if there are no close matches."


def match_entity(context, ent, candidates, llm):
    options = [f"{i + 1}: {lab_desc}" for i, lab_desc in enumerate(candidates)]
    options = '\n'.join(options)
    match_prompt = match_prompt_template.format(**{'context': context, 'options': options, 'text': ent})
    print(match_prompt)
    result = llm(match_prompt, n=1)  # TODO: revert this
    # result = llm(messages=[HumanMessage(content=match_prompt)])
    try:
        best_match_idx = int(re.search("\d+", result[0])[0]) - 1
        print(f"Matched:\n\t{ent}\n\t{candidates[best_match_idx]}")
    except ValueError:
        print(f"No match found: {result[0]}")
        best_match_idx = None
    except IndexError:
        print(f"Invalid option selected: {result[0]}")
        best_match_idx = None
    except TypeError:
        print(f"Invalid option selected: {result[0]}")
        best_match_idx = None

    return best_match_idx

def match_entities_musicbrainz(query, entities, model):
    matched_entities = {}
    for ent in entities:
        if len(ent) >= 300:
            continue
        candidates = get_candidates_for_entity(ent)
        candidate_strings = [c['phrase'] for c in candidates]
        ent_idx = match_entity(query, ent, candidate_strings, model)
        if ent_idx is None:
            continue
        selected_ent = candidates[ent_idx]
        qvalue = mb_ent_to_qent(selected_ent['id'])
        matched_entities[qvalue] = {
            'label': selected_ent['phrase'],
            'description': '',
            'musicbrainz': {
                'id': selected_ent['id'],
                'mbtype': selected_ent['type']
            }
        }
    return matched_entities

def get_candidates_for_entity(entity):
    try:
        label, description = entity.split(' - ')
    except ValueError:
        label = entity
        description = ""

    event_candidates = get_event_candidates(entity)
    recording_candidates = get_recording_candidates(entity)
    artist_candidates = get_artist_candidates(entity)
    work_candidates = get_work_candidates(entity)
    release_group_candidates = get_release_group_candidates(entity)
    place_candidates = get_place_candidates(entity)
    area_candidates = get_area_candidates(entity)
    candidates = artist_candidates +\
                 work_candidates +\
                 recording_candidates +\
                 event_candidates +\
                 release_group_candidates +\
                 place_candidates +\
                 area_candidates
    return candidates

def get_recording_candidates(label):
    candidates = []
    recordings = mbz.search_recordings(label, limit=10)['recording-list']
    recordings = filter_by_score(recordings)
    for recording in recordings:
        id = recording['id']
        recording_string = recording2string(recording)
        candidates.append({'id': id, 'phrase': recording_string, 'type': 'recording'})
    return candidates

def recording2string(recording):
    title = recording['title']
    try:
        artist_phrase = recording['artist-credit-phrase']
    except KeyError:
        artist_phrase = ""
        for artist in recording['artist-credit']:
            artist_phrase += artist['name'] + artist['joinphrase']
    try:
        disambiguation = f" ({recording['disambiguation']})"
    except KeyError:
        disambiguation = ""
    recording_string = f"Recording of {title} by {artist_phrase}{disambiguation}"
    return recording_string

def get_work_candidates(label):
    works = mbz.search_works(label, limit=10)['work-list']
    works = filter_by_score(works)
    candidates = []
    for work in works:
        id = work['id']
        try:
            work_string = work2string(work)
        except KeyError:
            continue
        candidates.append({'id': id, 'phrase': work_string, 'type': 'work'})
        # TODO: get main performers
    return candidates

def work2string(work):
    work_type = work['type']
    title = work['title']
    try:
        disambiguation = f" ({work['disambiguation']})"
    except KeyError:
        disambiguation = ""
    work_string = f"{title}{disambiguation} - {work_type} Written by "
    if 'artist-relation-list' in work:
        writers = [f"{artist['artist']['name']} ({artist['type']})" for artist in work['artist-relation-list']]
        writers_string = ", ".join(writers)
        work_string += writers_string
        # for artist in work['artist-relation-list']:
        #
        #     artists[artist['type']] = artists.get(artist['type'], []) + [artist['artist']['name']]
        # work_string += " ("
        # for artist_type, names in artists.items():
        #     work_string += artist_type + ": " + ", ".join(names)
        # work_string += ")"
    return work_string

def get_artist_candidates(label):
    candidates = []
    artists = mbz.search_artists(label, limit=10)['artist-list']
    artists = filter_by_score(artists)
    for artist in artists:
        id = artist['id']
        artist_string = artist2string(artist)

        candidates.append({'id': id, 'phrase': artist_string, 'type': 'artist'})
    return candidates

def artist2string(artist):
    artist_string = artist['name']
    artist_string += ' - Artist'
    if 'disambiguation' in artist:
        artist_string += f" ({artist['disambiguation']})"
    if 'sort-name' in artist:
        artist_string += f" ({artist['sort-name']})"
    return artist_string

def get_release_group_candidates(label):  # albums
    candidates = []
    # releases = mbz.search_releases(label, limit=10, format=['Digital Media', 'CD'], status='Official')
    release_groups = mbz.search_release_groups(label, limit=10, status='Official')['release-group-list']
    release_groups = filter_by_score(release_groups)
    for release_group in release_groups:
        id = release_group['id']
        try:
            release_group_string = release_group2string(release_group)
        except KeyError:
            continue
        candidates.append({'id': id, 'phrase': release_group_string, 'type': 'release-group'})
    return candidates

def release_group2string(release_group):
    title = release_group['title']
    release_type = release_group['primary-type'] or "Album/EP/Single"
    try:
        artist_phrase = release_group['artist-credit-phrase']
    except KeyError:
        artist_phrase = ""
        for artist in release_group['artist-credit']:
            artist_phrase += artist['name'] + artist['joinphrase']
    release_date = release_group['first-release-date']
    release_group_string = f"{title} - a {release_type} by {artist_phrase} (Released: {release_date})"
    return release_group_string

def release2string(release):
    title = release['title']
    release_date = release['date']
    release_country = release['country']
    release_group_string = f"Release of {title} - released in {release_country} on {release_date})"
    return release_group_string

def get_place_candidates(label):
    candidates = []
    places = mbz.search_places(label, limit=10)['place-list']
    places = filter_by_score(places)
    for place in places:
        id = place['id']
        try:
            place_string = place2string(place)
        except KeyError:
            continue
        candidates.append({'id': id, 'phrase': place_string, 'type': 'place'})
    return candidates

def place2string(place):
    name = place['name']
    place_type = place['type']
    area = place['area']['name']
    place_string = f"{name} - a place of type {place_type} in {area}"
    return place_string

def get_area_candidates(label):
    candidates = []
    areas = mbz.search_areas(label, limit=10)['area-list']
    areas = filter_by_score(areas)
    for area in areas:
        id = area['id']
        try:
            area_string = area2string(area)
        except KeyError:
            continue
        candidates.append({'id': id, 'phrase': area_string, 'type': 'area'})
    return candidates

def area2string(area):
    name = area['name']
    area_type = area['type']
    part_of = [a['area']['name'] for a in area['area-relation-list'] if a['type'] == 'part of'][0] if 'area-relation-list' in area else ""
    part_of = " in " + part_of if part_of else ""
    area_string = f"{name} - a {area_type}{part_of}"
    return area_string

def get_event_candidates(label):
    candidates = []
    events = mbz.search_events(label, limit=10)['event-list']
    events = filter_by_score(events)
    for event in events:
        try:
            id = event['id']
            event_string = event2string(event)
        except KeyError:
            continue
        candidates.append({'id': id, 'phrase': event_string, 'type': 'event'})
    return candidates

def event2string(event):
    name = event['name']
    event_type = event['type']
    try:
        disambiguation = f" ({event['disambiguation']})"
    except KeyError:
        disambiguation = ""
    date = event['life-span']['begin']
    event_string = f"{name}{disambiguation} - {event_type} on {date}"
    return event_string

def url2string(url):
    return url['resource']

def instrument2string(instrument):
    name = instrument['name']
    instrument_type = instrument['type']
    try:
        disambiguation = f" ({instrument['disambiguation']})"
    except KeyError:
        disambiguation = ""
    event_string = f"{name}{disambiguation} - instrument of type {instrument_type}"
    return event_string

def label2string(label):
    name = label['name']
    label_type = label['type']
    try:
        disambiguation = f" ({label['disambiguation']})"
    except KeyError:
        disambiguation = ""
    return f"{name}{disambiguation} - a label of type {label_type}"

def series2string(label):
    name = label['name']
    label_type = label['type']
    try:
        disambiguation = f" ({label['disambiguation']})"
    except KeyError:
        disambiguation = ""
    return f"{name}{disambiguation} - a series of type {label_type}"


def filter_by_score(search_results, filter=75):
    return [r for r in search_results if int(r['ext:score']) > filter]

def mb_prop_to_pprop(type_id):
    return 'P0' + re.sub(r'\D', '', type_id)[:4]

def mb_ent_to_qent(ent_id):
    return 'Q0' + re.sub(r'\D', '', ent_id)[:8]

def entity2string(entity, entity_type, with_request=False):
    if with_request:
        entity_id = entity['id']
        entity = get_data(entity_id, entity_type.replace('_', '-'))
        if entity is None:
            breakpoint()
    try:
        match entity_type:
            case 'work':
                return work2string(entity)
            case 'recording':
                return recording2string(entity)
            case 'artist':
                return artist2string(entity)
            case 'place':
                return place2string(entity)
            case 'area':
                return area2string(entity)
            case 'event':
                return event2string(entity)
            case 'release_group':
                return release_group2string(entity)
            case 'release':
                return release2string(entity)
            case 'url':
                return url2string(entity)
            case 'label':
                return label2string(entity)
            case 'series':
                return series2string(entity)
            case 'genre':
                return f"{entity['name']} - a genre of music"
    except KeyError as e:
        if with_request:
            breakpoint()
            raise e
        try:
            return entity2string(entity, entity_type, with_request=True)
        except mbz.musicbrainz.MusicBrainzError as mberror:
            breakpoint()
            raise mberror


if __name__=="__main__":
    # from langchain.chat_models import ChatOpenAI
    from pprint import pprint as pp
    candidates = get_candidates_for_entity("I don't know you - song by the Marias")
    # model = ChatOpenAI()
    # match = match_entities_musicbrainz("Which artists gave guitar support for the original recording of Blank Space by Taylor Swift?", ["Blank Space - song by taylor swift"], model)
