import copy
import logging
import re
from collections import Counter
from typing import List

from SPARQLWrapper import SPARQLExceptions
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

import musicbrainz.relation_options
from musicbrainz.initial_state import match_entities_musicbrainz, mb_prop_to_pprop
from musicbrainz.link_wikidata import link_musicbrainz_to_wikidata
import musicbrainz.expand_edges
from prompts.prompts_llama import EXPAND_KG_ENTITY_SELECTION_PROMPT, EXPAND_KG_RELATION_SELECTION_PROMPT, FULL_PROMPT, \
    EVALUATE_STATE_PROMPT, BASE_ACTION_SELECTION_PROMPT, THINK_PROMPT, ANSWER_PROMPT, EVALUATE_ANSWER_PROMPT
from wikidata.inital_state import parse_elements, match_entities_wikidata, \
    extract_entities_prompt_template
import wikidata.relation_options
import wikidata.expand_edges
from wikidata.link_musicbrainz import link_wikidata_to_musicbrainz

logging.basicConfig(level=logging.WARN)

class Actions:
    EXPAND_KG = 'EXPAND'
    ANSWER = 'ANSWER'
    THINK = 'THINK'
    # GENERATE_THOUGHT = 'GENERATE_THOUGHT'
    # GENERATE_ANSWER = 'GENERATE_ANSWER'
    SELECT_ENTITIES = 'SELECT_ENTITIES'
    SELECT_RELATION = 'SELECT_RELATION'

class Status:
    DONE = 'DONE'
    DEFAULT = 'DEFAULT'
    SELECTING_ENTITIES_TO_EXPAND = 'SELECTING_ENTITIES_TO_EXPAND'
    SELECTING_RELATION_TO_EXPAND = 'SELECTING_RELATION_TO_EXPAND'
    # ANSWERING = 'ANSWERING'
    # THINKING = 'THINKING'

DEFAULT_ACTION_CHAINS = [Actions.THINK, Actions.ANSWER, Actions.EXPAND_KG]



def status_transition(state, action):
    match state:
        case Status.DEFAULT:
            match action:
                case Actions.EXPAND_KG:
                    return Status.SELECTING_ENTITIES_TO_EXPAND
                case Actions.THINK:
                    return Status.DEFAULT
                case Actions.ANSWER:
                    return Status.DEFAULT
        case Status.SELECTING_ENTITIES_TO_EXPAND:
            return Status.SELECTING_RELATION_TO_EXPAND
        case Status.SELECTING_RELATION_TO_EXPAND:
            return Status.DEFAULT

ACTION_DESCRIPTIONS = {
    Actions.THINK: 'Generate relevant thoughts to solving the problem',
    Actions.EXPAND_KG: 'Expand the information in the knowledge graph through external APIs',
    Actions.ANSWER: 'Generate final answer once the problem is solved'
}

class WikidataTreeState:
    def __init__(self):
        self.trajectory = []
        self.entities: List[str] = []
        self.entity2kb = {}
        self.edges = []
        self.edge_dict = {}
        self.entity_groups = {}
        self.entity_labels = {}
        self.entity_descriptions = {}
        self.relations = {}  # {prop: prop_label}
        self.action_state = Status.DEFAULT
        self.cache = {}  # used to store data temporarily

    def add_entity(self, entity, label, description, kb2id_dict=None):
        self.entities.append(entity)
        self.entity_labels[entity] = label
        self.entity_descriptions[entity] = description
        if kb2id_dict:
            self.entity2kb[entity] = kb2id_dict

    def add_entity_group(self, entities):
        group = f"Group{len(self.entity_groups)}"
        self.entity_groups[group] = entities

    def add_edge(self, source, relation, object, attribute_string=""):
        self.edges.append((source, relation, object))
        if source not in self.edge_dict:
            self.edge_dict[source] = {}
        if relation not in self.edge_dict[source]:
            self.edge_dict[source][relation] = []
        self.edge_dict[source][relation].append((object, attribute_string))

    def add_relation(self, prop, prop_label):
        self.relations[prop] = prop_label

    def add_to_trajectory(self, text):
        self.trajectory.append(text)

    def transition_action_state(self, action):
        self.action_state = status_transition(self.action_state, action)

    def make_copy(self):
        child = copy.deepcopy(self)
        return child

    def kg_str(self):
        ent_str = "".join(
            [f"\t{e}: {self.entity_labels[e]} - {self.entity_descriptions[e]}\n" for e in self.entities]
        )
        edge_str = ""
        for s in self.edge_dict.keys():
            edge_str += f"{self.entity_labels[s] if s in self.entity_labels else s}:\n"
            for r in self.edge_dict[s].keys():
                edge_str += f"\t{self.relations[r]}: (count {len(self.edge_dict[s][r])})\n"
                object_strs = [f"\t\t{self.entity_labels[o] if o in self.entity_labels else o}{attr}\n" for o, attr in self.edge_dict[s][r]]
                edge_str += "".join(object_strs)

        # edge_str = "".join([f"\t({s}: {self.entity_labels[s]}, {self.relations[r]}, {o}: {self.entity_labels[o]})\n" for s, r, o in self.edges])

        group_str = ""
        for group_label, group in self.entity_groups.items():
            g_str = ", ".join(group)
            group_str += f"\t{group_label}: [{g_str}]\n"
        # return f"\nKG Entities:\n{ent_str}\nKG EDGES:\n{edge_str}\nKG Entity Groups:\n{group_str}"
        return f"\nKG Entities:\n{ent_str}\nKG EDGES:\n{edge_str}"

    def trajectory_str(self):
        return "\n".join(self.trajectory)


class StateTreeNode:
    def __init__(self, state: WikidataTreeState, parent=None, terminal=False, depth=0):
        self.state = state
        self.depth = depth
        self.parent = parent
        self.children: List[StateTreeNode] = []
        self.terminal = terminal
        self.order = None
        self.value = None

    def make_child(self, state, terminal=False):
        if self.terminal:
            raise ValueError("Cannot make a child of a terminal state")
        child = StateTreeNode(state, parent=self, depth=self.depth + 1, terminal=terminal)
        self.children.append(child)
        return child

    def set_terminal(self):
        self.terminal = True

    def set_value(self, value):
        self.value = value

    def traverse(self):
        yield self
        for c in self.children:
            yield from c.traverse()

    def __str__(self):
        return f"Node(order={self.order}, depth={self.depth}, parent={self.parent.order if self.parent is not None else 'root'}, value={self.value})"


class TreeOfTraversals:
    def __init__(self, llm=None, sample_breadth=1, max_depth=7, max_expansions=25, answer_threshold=0.8, knowledge_bases=['wikidata', 'musicbrainz']):
        self.state_tree: StateTreeNode = None
        self.query = None
        self.default_actions = None
        self.sample_breadth = sample_breadth
        self.max_depth = max_depth
        self.cur_node = None
        self.found_answer = False
        self.llm = llm
        self.expansion_count = 0
        self.answer_threshold = answer_threshold
        self.max_expansions = max_expansions
        self.knowledge_bases = knowledge_bases

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['llm']
        return state

    def run(self, query):
        self.expansion_count = 0
        self.query = query
        initial_state = self.construct_initial_state(query)
        self.state_tree = StateTreeNode(initial_state)
        self.state_tree.value = 1.0
        self.cur_node = None

        self.found_answer = False
        while not self.found_answer and self.expansion_count < self.max_expansions:
            self.step()
        if self.expansion_count >= self.max_expansions:
            print('Hit max expansions, stopping')
            if not self.answer_state():
                cur_node = self.select_node()
                answers = self.sample_answer(cur_node.state)
                for answer in answers:
                    new_state = cur_node.state.make_copy()
                    self.answer(new_state, answer)
                    child = cur_node.make_child(new_state, terminal=True)
                    value = self.evaluate_state(new_state)
                    child.set_value(value)
                    self.found_answer = True
            # attempt to generate an answer

        best_answer = self.answer_state()
        if best_answer is None:
            return "Max expansions. No answer"
        else:
            return best_answer.trajectory[-1].split(Actions.ANSWER)[-1].strip()

    def step(self):
        cur_node = self.select_node()
        if cur_node is None:
            print("No eligible states to expand in tree.")
            self.expansion_count = self.max_expansions  # TODO: Hacky way to end loop
            return
        cur_node.order = self.expansion_count
        sampled_actions = self.sample_actions(cur_node)
        self.expansion_count += 1
        sampled_actions = list(set(sampled_actions))  # filter exact duplicates
        new_state = None
        for argument in sampled_actions:
            new_state = cur_node.state.make_copy()
            terminal = False
            found_answer = False
            try:
                match cur_node.state.action_state:
                    case Status.DEFAULT:
                        self.default(new_state, argument)
                        if Actions.ANSWER in argument:
                            terminal = True
                            found_answer = True
                    case Status.SELECTING_ENTITIES_TO_EXPAND:
                        self.select_entities_sub_action(new_state, argument)
                    case Status.SELECTING_RELATION_TO_EXPAND:
                        self.select_relation_sub_action(new_state, argument)
                child = cur_node.make_child(new_state, terminal=terminal)
                value = self.evaluate_state(new_state)
                child.set_value(value)
                if found_answer and value >= self.answer_threshold:  # "definitely correct" threshold
                    self.found_answer = True
            except ValueError as e:
                print(f"ValueError: {str(e)}")
                self.invalid_action(new_state, e)
                # Does not add errors as children nodes
                # If no valid children then the node can be reselected

                # terminal = False  # TODO: Set to true but need to account for case where there is no nodes able to be expanded
                # child = cur_node.make_child(new_state, terminal=terminal)
                # child.set_value(0.0)
                # print(0.0)
        if not cur_node.children:
            if new_state is not None: # no error free children but there were children generated
                child = cur_node.make_child(new_state)  # add one error state to allow progress to continue
                child.set_value(-1.0)  # set value to -1 so it will not be repeated but also can still reach an answer
            else:  # no children generated (e.g. prompt too long)
                cur_node.set_value(-1.0)


    def initialize_state(self, query, data=None):  # TODO: Refactor and move to TreeOfThoughts
        if data is None:
            data = {}
        data['query'] = query
        result = self.llm(extract_entities_prompt_template.format(query=query), n=1, temperature=0.0)
        if 'Entities:' in result[0]:
            result[0] = result[0].split('Entities:')[1].strip()
        result[0] = result[0].split('\n\n')[0].split('\n \n')[0]
        data['entities'] = result[0]
        print(f"Extracted entities\n{data['entities']}")
        extracted_entities = parse_elements(data['entities'])
        matched_entities = {}
        wd2ent = {}
        unmatched_entities = extracted_entities
        if 'wikidata' in self.knowledge_bases:
            matched_entities, wd2ent, unmatched_entities = match_entities_wikidata(query, extracted_entities, self.llm)

        if 'musicbrainz' in self.knowledge_bases:
            to_search = set(unmatched_entities)
            ents2link = list(matched_entities.keys())
            # attempt to link entity if possible
            if ents2link:
                mblinks = link_wikidata_to_musicbrainz(ents2link)
                for wd_ent, mb_ent in mblinks.items():
                    matched_entities[wd_ent].update(mb_ent)
                linked_ents = set([wd2ent[wd_ent] for wd_ent in mblinks.keys()])
                # to_search = to_search.union(set(extracted_entities) - linked_ents)  # search for entities that were not linked
                to_search = to_search.union(set(extracted_entities))  # search for all entities

            mb_ents = match_entities_musicbrainz(query, to_search, self.llm)
            matched_entities.update(mb_ents)
        return matched_entities

    def construct_initial_state(self, query):
        initial_data = self.initialize_state(query)
        if not initial_data:
            initial_data = self.initialize_state(query)
        initial_state = WikidataTreeState()
        for key, value in initial_data.items():
            initial_state.add_entity(key, value['label'], value['description'], kb2id_dict=value)  # value contains other keys as well but also contains wikidata/musicbrainz
        initial_state.add_entity_group([k for k in initial_data.keys()])
        print(initial_state.kg_str())
        return initial_state

    def select_node(self) -> StateTreeNode:
        """selects the state to return based on value scores"""
        best_node = None
        best_value = -100000
        options = [node for node in self.state_tree.traverse()]
        options = sorted(options, key=lambda x: -x.depth)
        for node in options:
            if node.value > best_value and not node.terminal and not node.children:  # greater than or equal prioritizes depth when equal valued (faster)
                best_node = node
                best_value = node.value
        return best_node

    def sample_actions(self, cur_node: StateTreeNode):
        """samples next available actions for cur_state

        Uses policy llm response and parses to match action
        """
        results = None
        cur_state = cur_node.state
        if cur_state.action_state == Status.DEFAULT and cur_node.depth >= self.max_depth:
            results = self.sample_answer(cur_state)
            return [f"{Actions.ANSWER}: {r}" for r in results]
        match cur_state.action_state:
            case Status.DEFAULT:
                results = self.sample_default(cur_state)
            case Status.SELECTING_ENTITIES_TO_EXPAND:
                results = self.sample_select_entities_sub_action(cur_state)
            case Status.SELECTING_RELATION_TO_EXPAND:
                results = self.sample_select_relation_sub_action(cur_state)
        if results is None:
            breakpoint()
        return results

    def sample_k_outputs(self, prompt, stop='\n', oversample=None, match_pattern=None, max_tokens=None, use_match_as_action=True):
        print(prompt + "\n")
        k = oversample or self.sample_breadth
        results = self.llm(prompt, n=k, stop=stop, max_tokens=max_tokens)
        match_results = results
        if match_pattern is not None:
            match_results = [", ".join(re.findall(match_pattern, r)) for r in results]
        if oversample:
            counts = Counter(match_results)
            most_common = counts.most_common(n=self.sample_breadth)
            results = [r[0] for r in most_common]
        results = [r.strip() for r in results]
        print(results)
        return results

    def sample_default(self, state: WikidataTreeState):
        # Do not allow repeat thoughts
        options = DEFAULT_ACTION_CHAINS[:]
        if state.trajectory and 'THINK' in state.trajectory[-1]:
            options.remove(Actions.THINK)
            options = "EXPAND/ANSWER: <answer>"
        else:
            options = "THINK: <thought>/EXPAND/ANSWER: <answer>"
        # options = ", ".join(DEFAULT_ACTION_CHAINS)
        current_prompt = BASE_ACTION_SELECTION_PROMPT.format(options=options)
        full_prompt = self.construct_full_prompt(current_prompt, state)
        actions = self.sample_k_outputs(full_prompt, match_pattern=("(THINK|EXPAND|ANSWER)"), use_match_as_action=False, max_tokens=100)
        return actions

    def default(self, state: WikidataTreeState, argument):
        for action_choice in DEFAULT_ACTION_CHAINS:  # need to match one of the options
            if re.match(f"{action_choice}.*", argument.strip()) is not None:
                arg = action_choice.join(argument.split(action_choice)[1:])  # only split the first action selection.
                if action_choice == Actions.THINK:  # THINK requires no additional action samples
                    arg = arg.split('ANSWER:')[0]
                    self.think(state, arg)
                elif action_choice == Actions.ANSWER:  # ANSWER requires no additional action samples
                    self.answer(state, arg)
                elif action_choice == Actions.EXPAND_KG:
                    arg = arg.split('ANSWER:')[0]
                    self.expand_kg(state, arg)
                return

        raise ValueError(f"'{argument}' is not a valid action to pick.")

    def sample_think(self, state: WikidataTreeState):
        #TODO: Remove this?
        raise DeprecationWarning('Should not need to sample thoughts separately')
        full_prompt = self.construct_full_prompt(THINK_PROMPT, state)
        return self.sample_k_outputs(full_prompt)

    def think(self, state: WikidataTreeState, argument):
        obs = f"THINK: {argument}"
        state.add_to_trajectory(obs)
        state.transition_action_state(Actions.THINK)

    def sample_answer(self, state: WikidataTreeState):
        full_prompt = self.construct_full_prompt(ANSWER_PROMPT, state)
        return self.sample_k_outputs(full_prompt, stop=None, max_tokens=256)

    def answer(self, state: WikidataTreeState, arguments):
        if Actions.ANSWER + ":" in arguments:
            arguments = arguments.split(Actions.ANSWER + ":")[1].strip()
        obs = f"{Actions.ANSWER}: {arguments}"
        state.add_to_trajectory(obs)
        state.transition_action_state(Actions.ANSWER)

    def expand_kg(self, state: WikidataTreeState, arguments):
        state.add_to_trajectory(f"{Actions.EXPAND_KG}: {arguments}")
        state.transition_action_state(Actions.EXPAND_KG)

    def select_db(self, state: WikidataTreeState):
        raise NotImplementedError

    def sample_select_entities_sub_action(self, state: WikidataTreeState):
        # options = state.entities + [group_label for group_label in state.entity_groups.keys()]
        options = state.entities
        current_prompt = EXPAND_KG_ENTITY_SELECTION_PROMPT.format(options=', '.join(options))
        full_prompt = self.construct_full_prompt(current_prompt, state)
        options_sorted = sorted(options, key=len, reverse=True)
        pattern = "(" + "|".join(options_sorted) + ")"
        oversample = 1 if self.sample_breadth == 1 else self.sample_breadth * 2
        return self.sample_k_outputs(full_prompt, oversample=oversample, match_pattern=pattern, max_tokens=128)

    def select_entities_sub_action(self, state: WikidataTreeState, argument):
        selected_entities = []
        for ent in state.entities:
            if ent.casefold() in argument.casefold():
                selected_entities.append(ent)
        # for group in state.entity_groups.keys():
        #     if group.casefold() in argument.casefold():
        #         selected_entities += state.entity_groups[group]
        # if "group" in argument.lower():
        #     group_number = int(argument.lower().split("group")[1])
        #     selected_entities = state.entity_groups[group_number][1]
        # else:
        #     selected_entities = []
        #     possible_entities = argument.split()
        #     for ent in possible_entities:
        #         ent = re.sub(r'\W+', '', ent)
        #         if ent in state.entities:
        #             selected_entities.append(ent)
        if not selected_entities:
            raise ValueError(f'Could not parse any entities from {argument}')
        state.cache['selected_entities'] = selected_entities
        state.add_to_trajectory(f"{Actions.SELECT_ENTITIES}: {selected_entities}")
        state.transition_action_state(Actions.SELECT_ENTITIES)

    @staticmethod
    def extract_entities_in_kb(state, entities, kb):
        entity_keys = [e for e in entities if kb in state.entity2kb[e]]
        kb_entity = [state.entity2kb[e][kb] for e in entities if kb in state.entity2kb[e]]
        return entity_keys, kb_entity

    def sample_select_relation_sub_action(self, state: WikidataTreeState):
        prop2trueprop = {}
        backward_props, backward_labels, forward_props, forward_labels = [], [], [], []
        if 'wikidata' in self.knowledge_bases:
            entity_keys, wikidata_entities = self.extract_entities_in_kb(state, state.cache['selected_entities'], 'wikidata')
            if wikidata_entities:
                forward_props, forward_labels = wikidata.relation_options.get_forward_relations(wikidata_entities)
                try:
                    backward_props, backward_labels = wikidata.relation_options.get_backward_relations(wikidata_entities)
                except SPARQLExceptions.EndPointInternalError as e:
                    logging.warning(f"Could not get backward edges for {wikidata_entities}", exc_info=e)
                    backward_props, backward_labels = [], []
                for p in forward_props:
                    prop2trueprop[p] = {'wikidata': p}
                for p in backward_props:
                    prop2trueprop['~' + p] = {'wikidata': p}
        if 'musicbrainz' in self.knowledge_bases:
            entity_keys, musicbrainz_entities = self.extract_entities_in_kb(state, state.cache['selected_entities'], 'musicbrainz')
            for mbent in musicbrainz_entities:
                mbprops, mblabels = musicbrainz.relation_options.get_relations(mbent['id'], mbent['mbtype'])
                mbprops_text = [mb_prop_to_pprop(p) for p in mbprops]
                forward_props += mbprops_text
                forward_labels += mblabels
                for p_text, p_true in zip(mbprops_text, mbprops):
                    prop2trueprop[p_text] = {'musicbrainz': p_true}

        forward_options = '\n'.join([f"{p}: {l}" for p, l in zip(forward_props, forward_labels)])
        backward_options = '\n'.join([f"~{p}: is {l} of" for p, l in zip(backward_props, backward_labels)])
        state.cache['forward_props'] = forward_props
        state.cache['forward_prop_labels'] = forward_labels
        state.cache['backward_props'] = backward_props
        state.cache['backward_prop_labels'] = backward_labels
        state.cache['prop2trueprop'] = prop2trueprop
        selected_entities = [state.entity_labels[e] for e in state.cache['selected_entities']]
        current_prompt = EXPAND_KG_RELATION_SELECTION_PROMPT.format(selected_entities=selected_entities,
                                                                    outgoing=forward_options,
                                                                    incoming=backward_options)
        full_prompt = self.construct_full_prompt(current_prompt, state)
        oversample = 1 if self.sample_breadth == 1 else self.sample_breadth*3
        res = self.sample_k_outputs(full_prompt, oversample=oversample, match_pattern="~?P\d+", max_tokens=10)
        return res

    def select_relation_sub_action(self, state: WikidataTreeState, argument):
        try:
            full_prop = re.search("~?P\d+", argument)[0]  # TODO: Change search pattern to use the list of relations available (all cases)
        except TypeError:
            raise ValueError(f'Unable to extract property PID from {argument}')
        incoming_edge = (full_prop[0] == '~')

        if not incoming_edge and full_prop not in state.cache['forward_props']:
            incoming_edge = True
            full_prop = '~' + full_prop

        if incoming_edge:
            prop = re.search("P\d+", full_prop)[0]
            prop_idx = state.cache['backward_props'].index(prop)
            prop_label = state.cache['backward_prop_labels'][prop_idx]
        else:
            incoming_edge = False
            prop = full_prop
            prop_idx = state.cache['forward_props'].index(full_prop)
            prop_label = state.cache['forward_prop_labels'][prop_idx]

        new_ent_count = 0
        if 'wikidata' in self.knowledge_bases and 'wikidata' in state.cache['prop2trueprop'][full_prop]:
            using = 'wikidata'
            entity_keys, kb_entities = self.extract_entities_in_kb(state, state.cache['selected_entities'], 'wikidata')
            try:
                sources, relations, objects, attributes, new_entities = wikidata.expand_edges.get_edges(kb_entities, prop, incoming_edge)
            except SPARQLExceptions.EndPointInternalError:
                raise ValueError("Query could not be run.")
            sources = [entity_keys[kb_entities.index(e)] for e in sources]
            kb2id_dict = {e: {'wikidata': e} for e in new_entities.keys()}  # entity id is wikidata QID
        elif 'musicbrainz' in self.knowledge_bases and 'musicbrainz' in state.cache['prop2trueprop'][full_prop]:
            using = 'musicbrainz'
            entity_keys, kb_entities = self.extract_entities_in_kb(state, state.cache['selected_entities'], 'musicbrainz')
            sources, relations, objects, attributes, new_entities = [], [], [], [], {}
            for entity_key, mbent in zip(entity_keys, kb_entities):
                _sources, _relations, _objects, _attributes, _new_entities = musicbrainz.expand_edges.get_edges(mbent['id'],
                                                                                                            mbent['mbtype'],
                                                                                                            state.cache['prop2trueprop'][prop]['musicbrainz'])
                _sources = [entity_key] * len(_relations)
                sources += _sources
                relations += _relations
                objects += _objects
                attributes += _attributes
                new_entities.update(_new_entities)
            kb2id_dict = {e: new_entities[e] for e in new_entities.keys()}  # includes label and description but that's okay for now
        else:
            raise Exception(f"Unsure which kg to use for {state.cache['selected_entities']} and {full_prop}")

        # link wikidata to musicbrainz
        if using == 'wikidata' and 'musicbrainz' in self.knowledge_bases:
            if new_entities:
                wd2mb = link_wikidata_to_musicbrainz(list(new_entities.keys()))
                for ent in wd2mb.keys():
                    kb2id_dict[ent].update(wd2mb[ent])

        if using == 'musicbrainz' and 'wikidata' in self.knowledge_bases:
            for k, mbent in kb2id_dict.items():
                qid = link_musicbrainz_to_wikidata(mbent['musicbrainz']['id'], mbent['musicbrainz']['mbtype'])
                if qid is not None:
                    kb2id_dict[k].update({'wikidata': qid})

        for key in new_entities.keys():
            if key not in state.entities:  # new entities might be duplicates, need to check
                state.add_entity(key, new_entities[key]['label'], new_entities[key]['description'], kb2id_dict=kb2id_dict[key])
                new_ent_count += 1

        state.add_entity_group([k for k in new_entities.keys()])  # add all entities to entity group

        if incoming_edge:
            full_prop_label = f"is {prop_label} of"
            state.add_relation(full_prop, full_prop_label)
            state.add_to_trajectory(f"{Actions.SELECT_RELATION}: {full_prop} - {full_prop_label}")
        else:
            full_prop_label = f"has {prop_label}"
            state.add_relation(full_prop, full_prop_label)
            state.add_to_trajectory(f"{Actions.SELECT_RELATION}: {full_prop} - {full_prop_label}")

        # Add edges to state
        new_edge_count = 0
        new_edges_added_for = Counter()
        for s, r, o, a in zip(sources, relations, objects, attributes):
            if (s,r,o) not in state.edges:
                new_edge_count += 1
                state.add_edge(s, full_prop, o, attribute_string=a)
                new_edges_added_for[s] += 1

        # union = []
        # for new_ent in new_entities:
        #     r = relations[0]
        #     in_union = True
        #     for e in state.cache['selected_entities']:
        #         to_find = (new_ent,r,e) if incoming_edge else (e,r,new_ent)
        #         if to_find not in zip(subjects, relations, objects):
        #             in_union = False
        #     if in_union:
        #         union.append(new_ent)

        if not new_edge_count and not new_ent_count:
            raise ValueError("Action resulted in no new entities or edges being added")

        state.add_to_trajectory(f"Observation: {new_ent_count} new entities, {new_edge_count} new edges added")

        state.transition_action_state(Actions.SELECT_RELATION)

    def invalid_action(self, state: WikidataTreeState, exception):
        state.add_to_trajectory("ERROR: " + str(exception))
        state.action_state = Status.DEFAULT

    def construct_full_prompt(self, current_prompt, state):
        full_prompt = FULL_PROMPT.format(query=self.query,
                                         kg_state=state.kg_str(),
                                         trajectory=state.trajectory_str(),
                                         current_prompt=current_prompt)
        return full_prompt

    def evaluate_state(self, state):
        if Actions.ANSWER in state.trajectory[-1]:
            query = self.query
            kg_state = state.kg_str()
            answer = self.answer_from_state(state)
            full_prompt = EVALUATE_ANSWER_PROMPT.format(query=query, answer=answer, kg_state=kg_state)
            print(full_prompt)
            result = self.llm(full_prompt, stop=None, n=1, temperature=0)[0]
            print(result)
        else:
            current_prompt = EVALUATE_STATE_PROMPT
            full_prompt = self.construct_full_prompt(current_prompt, state)
            result = self.llm(full_prompt, stop="\n", n=1, temperature=0)
            if result == []:
                return 0.0
            else:
                result = result[0]
            print(result)
        value_match = re.search("\d\.\d+", result)
        if value_match is None:
            print(f"Could not get value of '{result}'")
            value = 0.0
        else:
            value = float(value_match[0])
        return value

    def call_model(self, prompt, stop="\n"):
        if isinstance(self.llm, ChatOpenAI):
            return self.llm([HumanMessage(content=prompt)], stop=[stop])
        else:
            return self.llm(prompt, stop=stop)

    def answer_state(self):
        # gets "best" answer state
        answer_nodes = []
        for node in self.state_tree.traverse():
            if len(node.state.trajectory) and Actions.ANSWER in node.state.trajectory[-1]:
                answer_nodes.append(node)
        if not answer_nodes:
            return None
        best_node = max(answer_nodes, key=lambda x: x.value)
        best_value = best_node.value
        # best_node = random.choice([a for a in answer_nodes if a.value == best_value])  # TODO: remove randomness here (can cause inconsistency when analyzing answer state)
        best_nodes = [a for a in answer_nodes if a.value == best_value]
        best_nodes = sorted(best_nodes, key=lambda x: x.depth)
        best_node = best_nodes[-1]
        return best_node.state

    @staticmethod
    def answer_from_state(state):
        return state.trajectory[-1].split(Actions.ANSWER)[-1].strip()


