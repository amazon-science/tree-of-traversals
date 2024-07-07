# Tree of Traversals

This repo contains the code for Tree of Traversals (paper to come)

## Setup 

### Environment

`conda env create -f environment.yml`
(works in testing)

### Wikimapper (deprecated)

Wikimapper is no longer needed. If this gives any challenges (it shouldn't) feel free to remove from codebase.

### OpenAI

Set the environment variable `OPENAI_API_KEY` to your openai api key. Or use any other tool recognized by the openai python package.

## Run the model

To run the model with a single query, use

`python main.py --query="<Your Query Here>"`

Options include

`--sample_breadth` Set sample breadth. Set to 1 for Chain of Traversals

`--max_depth` Depth at which model is forced to give an answer. Can complete ExpandKG action before doing so)

`--llm` Select which LLM to use. Currently supports OpenAI chat models. Unlikely to work with open source models even though there is code to support it (future work).

`--knowledge_bases` Select which knowledge bases to include. Currently only supports `'wikidata'` and `'musicbrainz'`.
    `--knowledge_bases wikidata musicbrainz` (to use both)

Warning: Expect musicbrainz to be slow due to rate limits on the public API endpoint. 
If you have access to higher rate limits or are changing the endpoint to one without rate limits, then remove the `sleep` from `musicbrainz/relation_options.py:27 (get_data)`.
MusicBrainz `user-agent` is set in both `musicbrainz/initial_state` as well as in `musicbrainz/relation_options`.

## Running experiments

### 2WikiMultiHop

To run Tree of Traversals (breadth 3, depth 7):
`python wikimultihop.py --results_file results/<llm>/2wiki/ToTraversals_b3_d7.json --sample_breadth=3 --max_depth=7 --llm=<llm>`

Note that `results_file` uses caching. If you start a run and need to interrupt, or it hangs, or needs to be restarted for any reason. It will maintain the prior results. Likewise, be careful if you make changes, you will need to rename or change the results_file
To run Chain of Traversals (breadth 1, depth 7):

`python wikimultihop.py --results_file results/<llm>/2wiki/CoTraversals_d7.json --sample_breadth=1 --max_depth=7 --llm=<llm>`

To run Chain of Thought:

`python wikimultihop.py --results_file results/<llm>/2wiki/CoThoughts.json --model='chain-of-thoughts' --llm=<llm>`

### QALD10

Tree of Traversals:

`python qald.py --results_file results/<llm>/qald/ToTraversals_b3_d7.json --sample_breadth=3 --max_depth=7 --llm=<llm>`

Chain of Traversals:

`python qald.py --results_file results/<llm>/qald/CoTraversals_d7.json --sample_breadth=1 --max_depth=7 --llm=<llm>`

Chain of Thought:

`python qald.py --results_file results/<llm>/qald/CoThought.json --model=chain-of-thoughts --llm=<llm>`

### Music_x_Wiki

Tree of Traversals:
`python wiki_x_music.py --results_file results/<llm>/wiki_x_music/ToTraversals_b3_d7.json --sample_breadth=3 --max_depth=7 --llm=<llm>`

Chain of Traversals:
`python wiki_x_music.py --results_file results/<llm>/wiki_x_music/CoTraversals_d7.json --sample_breadth=1 --max_depth=7 --llm=<llm>`

Chain of Thought:
`python wiki_x_music.py --results_file results/<llm>/wiki_x_music/CoThought.json --model=chain-of-thoughts --llm=<llm>`

## Choosing an LLM

Setting the parameter `--llm=<model_id>` changes what LLM is used for inference.

Options include Bedrock models from Anthropic, Meta, or Cohere as well as any custom SageMaker endpoint.

## Prompts

There are multiple prompt files available for use under the prompts directory. You may change the prompts being used by
changing the `from prompts.prompts_*` import line in `treeoftraversals.py` to use the correct file.

E.g. For Llama set it to `from prompts.prompts_llama import ...`

## Questions

Reach out to Elan Markowitz (elanmark@amazon.com)
