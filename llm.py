import copy

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

from langchain import HuggingFacePipeline

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

from retry import retry
import json
import openai
import logging

import bedrock
import botocore

OPENAI_ERRORS = (openai.error.Timeout, openai.error.ServiceUnavailableError, openai.error.APIError, openai.error.APIConnectionError)


class OPENAI_LLM:
    def __init__(self, model_id, n=1):
        self.model_id = model_id
        self.n = n

    @retry(openai.error.RateLimitError, tries=2, delay=30)
    @retry(OPENAI_ERRORS, tries=5, backoff=2)
    def __call__(self, prompt, n=None, stop=None, temperature=1.1, max_tokens=None):
        n = n or self.n
        try:
            result = openai.ChatCompletion.create(
                model=self.model_id,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                n=n,
                stop=stop,
                temperature=temperature,
                max_tokens=max_tokens
            )
        except openai.error.InvalidRequestError:
            logging.warning(f"Prompt too long: {prompt}")
            return []
        result = [choice['message']['content'] for choice in result['choices']]
        return result

class BedrockLLM:
    boto3_bedrock = bedrock.get_bedrock_client(region='us-east-1')  # class variable so we can pickle the LLM without the boto session.
    
    def __init__(self, model_id, n=1):
        self.model_id = model_id
        self.accept = 'application/json'
        self.contentType = 'application/json'
        self.n = n
        self.default_temp = 1.0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def reset_counts(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def get_payload(self, prompt, n, stop, temperature, max_tokens):
        payload = {
            "prompt": "Human:\n" + prompt + "\nAssistant:",
            "max_tokens_to_sample": 4096  # requires max_tokens_to_sample
        }
        if stop:
            payload['stop_sequences'] = [stop] if isinstance(stop, str) else stop
        if temperature is None:
            payload['temperature'] = self.default_temp
        elif temperature > 0.0:
            payload['temperature'] = temperature
        else:
            payload['temperature'] = 0
        if max_tokens:
            payload['max_tokens_to_sample'] = max_tokens
        return payload
    
    def __call__(self, prompt, n=None, stop=None, temperature=None, max_tokens=None):
        n = n or self.n
        result = []
        payload = self.get_payload(prompt, n, stop, temperature, max_tokens)
        for i in range(n):
            response = self.boto3_bedrock.invoke_model(body=json.dumps(payload), 
                                                       modelId=self.model_id, 
                                                       accept=self.accept, 
                                                       contentType=self.contentType)
            response_body = json.loads(response.get('body').read())
            res = response_body['completion']
            result.append(res)
        return result


class AnthropicBedrockLLM(BedrockLLM):
    def __init__(self, model_id, n=1):
        super().__init__(model_id, n)
        self.default_temp = 1.0


class Llama2Bedrock(BedrockLLM):
    def get_payload(self, prompt, n, stop, temperature, max_tokens):
        payload = {
            "prompt": f"[INST] {prompt} [/INST]\n\n",
            "max_gen_len": 128,  # requires max_tokens_to_sample
        }
        # if stop:
        #     payload['stop_sequences'] = [stop] if isinstance(stop, str) else stop
        if temperature is None:
            payload['temperature'] = self.default_temp
        elif temperature > 0.0:
            payload['temperature'] = temperature
        if max_tokens:
            payload['max_gen_len'] = max_tokens
        return payload

    def __call__(self, prompt, n=None, stop=None, temperature=None, max_tokens=None):
        n = n or self.n
        result = []
        payload = self.get_payload(prompt, n, stop, temperature, max_tokens)
        for i in range(n):
            response = self.boto3_bedrock.invoke_model(body=json.dumps(payload), 
                                                       modelId=self.model_id, 
                                                       accept=self.accept, 
                                                       contentType=self.contentType)
            response_body = json.loads(response.get('body').read())
            self.prompt_tokens += response_body['prompt_token_count']
            self.completion_tokens += response_body['generation_token_count']
            text = response_body['generation'].strip()
            if stop is not None:
                if isinstance(stop, str):
                    text = text.split(stop)[0]
                else:
                    for stopper in stop:
                        text = text.split(stopper)[0]
            result.append(text)
        return result


class TitanBedrock(BedrockLLM):
    def get_payload(self, prompt, n, stop, temperature, max_tokens):
        payload = {
            "prompt": f"{prompt}",
            "max_gen_len": 128,  # requires max_tokens_to_sample
        }
        # if stop:
        #     payload['stop_sequences'] = [stop] if isinstance(stop, str) else stop
        if temperature is None:
            payload['temperature'] = self.default_temp
        elif temperature > 0.0:
            payload['temperature'] = temperature
        if max_tokens:
            payload['max_gen_len'] = max_tokens
        return payload

    def __call__(self, prompt, n=None, stop=None, temperature=None, max_tokens=None):
        n = n or self.n
        result = []
        payload = self.get_payload(prompt, n, stop, temperature, max_tokens)
        for i in range(n):
            response = self.boto3_bedrock.invoke_model(body=json.dumps(payload), 
                                                       modelId=self.model_id, 
                                                       accept=self.accept, 
                                                       contentType=self.contentType)
            response_body = json.loads(response.get('body').read())
            self.prompt_tokens += response_body['prompt_token_count']
            self.completion_tokens += response_body['generation_token_count']
            text = response_body['generation'].strip()
            if stop is not None:
                if isinstance(stop, str):
                    text = text.split(stop)[0]
                else:
                    for stopper in stop:
                        text = text.split(stopper)[0]
            result.append(text)
        return result


class CohereBedrockLLM(BedrockLLM):
    def __init__(self, model_id, n=1):
        super().__init__(model_id, n)
        self.default_temp = 1.1

    def get_payload(self, prompt, n, stop, temperature, max_tokens):
        payload = {
            "prompt": "Human:\n" + prompt + "\nAssistant:",
            "max_tokens": 4096,  # requires max_tokens_to_sample
            "num_generations": n
        }
        if stop:
            payload['stop_sequences'] = [stop] if isinstance(stop, str) else stop
        if temperature is None:
            payload['temperature'] = self.default_temp
        elif temperature > 0.0:
            payload['temperature'] = temperature
        if max_tokens:
            payload['max_tokens'] = max_tokens
        return payload

    def __call__(self, prompt, n=None, stop=None, temperature=None, max_tokens=None):
        n = n or self.n
        result = []
        payload = self.get_payload(prompt, n, stop, temperature, max_tokens)
        remainder = 0
        if payload["num_generations"] > 5:
            payload["num_generations"] = 5
            remainder = n - payload["num_generations"]
        response = self.boto3_bedrock.invoke_model(body=json.dumps(payload),
                                                   modelId=self.model_id,
                                                   accept=self.accept,
                                                   contentType=self.contentType)
        response_body = json.loads(response.get('body').read())
        result = [g['text'] for g in response_body['generations']]
        if remainder > 0:
            return result + self(prompt, remainder, stop, temperature, max_tokens)
        return result


class SagemakerLLM:
    def __init__(self, endpoint, n=1):
        self.endpoint = endpoint
        self.predictor = Predictor(endpoint)
        self.predictor.serializer = JSONSerializer()
        self.predictor.deserializer = JSONDeserializer()
        self.n = n
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def reset_counts(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @retry(botocore.errorfactory.ClientError, tries=5, backoff=2)
    def __call__(self, prompt, n=None, stop=None, temperature=1.0, max_tokens=128):
        n = n or self.n
        result = []
        # full_prompt = f"[INST] {prompt} [/INST]"
        full_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
        if "openchat" in self.endpoint:
            full_prompt = f"GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant:"
        if "tulu" in self.endpoint:
            full_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
        payload = {
            "inputs": full_prompt,
            "parameters": {
                "do_sample": True,
                "return_full_text": False,
                "stop": ["</s>"]
            }
        }
        if stop:
            payload['parameters']['stop'] = ["</s>", stop] if isinstance(stop, str) else stop + ["</s>"]
        if temperature:
            payload['parameters']['temperature'] = temperature
        if max_tokens:
            payload['parameters']['max_new_tokens'] = max_tokens
        for i in range(n):
            res = self.predictor.predict(payload)
            text = res[0]['generated_text']
            for stopper in payload['parameters']['stop']:
                text = text.split(stopper)[0]
            result.append(text)
        return result

    
class HUGGINGFACE_LLM:
    def __init__(self, model_id, n=1):
        self.n = n
        self.pipe = self.create_pipe(model_id, n=n)

    def create_pipe(self, model_id):
        model_id = model_id
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # model_4bit = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto",
        #                                                   trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            use_cache=True,
            device_map="auto",
            max_new_tokens=20,
            do_sample=True,
            top_k=1,
            num_return_sequences=self.n,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            trust_remote_code=True
        )
        return pipe

    def __call__(self, prompt, n=None):
        n = n or self.n
        result = self.pipe(prompt)
        return result


def get_llm(model_id, n=1):
    if model_id in ["gpt-3.5-turbo", "gpt4"] or "gpt-3.5-turbo" in model_id:
        return OPENAI_LLM(model_id, n=n)
    elif 'anthropic' in model_id:
        return AnthropicBedrockLLM(model_id, n=n)
    elif 'cohere' in model_id:
        return CohereBedrockLLM(model_id, n=n)
    elif 'llama' in model_id:
        return Llama2Bedrock(model_id, n=n)
    else:
        return SagemakerLLM(model_id, n=n)

# MODEL_ID = 'gpt-3.5-turbo'
# LLM_MODEL = get_llm(MODEL_ID, None)

#
# if __name__ == "__main__":
#     LLM_MODEL = get_llm(MODEL_ID, None)
#     result = LLM_MODEL([HumanMessage(content="Who is the matrix made for?")])
#     print(result.content)
