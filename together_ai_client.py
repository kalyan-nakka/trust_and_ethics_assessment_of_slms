from typing import List, Dict, Any, Optional, Union, Set

import requests
from retrying import retry

from helm.common.cache import Cache, CacheConfig
from helm.common.request import Request, RequestResult, Sequence, Token
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
)
from helm.proxy.clients.client import Client, wrap_request_time, truncate_sequence, cleanup_str
from helm.proxy.models import Model, TEXT_MODEL_TAG, MODEL_NAME_TO_MODEL, ALL_MODELS
from helm.common.hierarchical_logger import hlog

_ASYNC_MODELS: Set[str] = {
    # Legacy models
    "alpaca-7b",
    "pythia-7b",
    "vicuna-13b",
    # Production models
    "redpajama-incite-base-3b-v1",
    "redpajama-incite-instruct-3b-v1",
    "redpajama-incite-base-7b",
    "redpajama-incite-instruct-7b",
    "dolly-v2-3b",
    "dolly-v2-7b",
    "dolly-v2-12b",
    "llama-7b",
    "llama-13b",
    "llama-30b",
    "llama-65b",
    "llama-2-7b",
    "llama-2-13b",
    "llama-2-70b",
    "pythia-1b-v0",
    "pythia-2.8b-v0",
    "pythia-6.9b",
    "pythia-12b-v0",
    "stablelm-base-alpha-3b",
    "stablelm-base-alpha-7b",
}
"""Together models to use async requests for.

Currently async requests are only used for models that are timing out,
because async requests are slower than sync requests.

Note: These should be HELM model names, not Together model name aliases."""


MODEL_ALIASES: Dict[str, str] = {
    # Legacy models
    "flan-t5-xxl": "flan-t5-xxl-hf",
    "h3-2.7b": "h3-2.7b-h3",
    "opt-1.3b": "opt-1.3b-ft-tp1",
    "opt-6.7b": "opt-6.7b-ft-tp1",
    # Together's models are half-precision are default,
    # and the full-precision models are suffixed e.g.
    # alpaca-7b is half-precision
    # alpaca-7b-full-precision is full-precision
    "alpaca-7b": "alpaca-7b-full-precision",
    "pythia-7b": "pythia-7b-full-precision",
    "vicuna-13b": "vicuna-13b-full-precision",
    # Production models
    "redpajama-incite-base-3b-v1": "togethercomputer/RedPajama-INCITE-Base-3B-v1",
    "redpajama-incite-instruct-3b-v1": "togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
    "redpajama-incite-base-7b": "togethercomputer/RedPajama-INCITE-7B-Base",
    "redpajama-incite-instruct-7b": "togethercomputer/RedPajama-INCITE-7B-Instruct",
    "dolly-v2-3b": "databricks/dolly-v2-3b",
    "dolly-v2-7b": "databricks/dolly-v2-7b",
    "dolly-v2-12b": "databricks/dolly-v2-12b",
    "llama-7b": "huggyllama/llama-7b",
    "llama-13b": "huggyllama/llama-13b",
    "llama-30b": "huggyllama/llama-30b",
    "llama-65b": "huggyllama/llama-65b",
    "llama-2-7b": "togethercomputer/llama-2-7b",
    "llama-2-13b": "togethercomputer/llama-2-13b",
    "llama-2-70b": "togethercomputer/llama-2-70b",
    "pythia-1b-v0": "EleutherAI/pythia-1b-v0",
    "pythia-2.8b-v0": "EleutherAI/pythia-2.8b-v0",
    "pythia-6.9b": "EleutherAI/pythia-6.9b",
    "pythia-12b-v0": "EleutherAI/pythia-12b-v0",
    "stablelm-base-alpha-3b": "stabilityai/stablelm-base-alpha-3b",
    "stablelm-base-alpha-7b": "stabilityai/stablelm-base-alpha-7b",
    
    # AF Interested Models
    "gemma": "google/gemma",
    "phi-2": "microsoft/phi-2",
    "RedPajama-INCITE-Chat-3B-v1": "togethercomputer/RedPajama-INCITE-Chat-3B-v1",
    "Llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
    "Mistral-7B-Instruct-v0.2": "mistralai/Mistral-7B-Instruct-v0.2",
    "Qwen1.5-7B-Chat": "Qwen/Qwen1.5-7B-Chat",
}
"""Together model name aliases.

HELM users use a shorter model name (e.g. together/flan-t5-xxl)
whereas the Together client sends and caches requests using
a longer model name that is suffixed with the implementation framework
(e.g. flan-t5-xxl-hf). This allows tracking exactly which
implementation was used in the cached results, since some results may
be different depending on the implementation (e.g. efficiency metrics).
This also allows future migration of results in the case of changes of
available implementations on Together."""


class TogetherAIClientError(Exception):
    pass


class JobNotFinishedError(TogetherAIClientError):
    """Exception raised when trying to get a response for a Together async job that has not finished"""

    pass


class TogetherAIClient(Client):
    """
    Client for the models where we evaluate offline. Since the queries are handled offline, the `TogetherClient` just
    checks if the request/result is cached. We return the result if it's in the cache. Otherwise, we return an error.
    """

    CHAT_ENDPOINT: str = "https://api.together.xyz/v1/chat/completions"
    COMPLETION_ENDPOINT: str = "https://api.together.xyz/v1/completions"
    RETRIEVE_JOB_MAX_WAIT_SECONDS: int = 60

    @staticmethod
    def convert_to_raw_chat_request(request: Request) -> Dict:
        # Following the examples from https://github.com/togethercomputer/open-models-api
        return {
            "temperature": request.temperature,
            "n": request.num_completions,
            "max_tokens": request.max_tokens,
            "stop": request.stop_sequences or None,
            "echo": request.echo_prompt,
            "top_p": request.top_p,
            "top_k": request.top_k_per_token,
            "messages": request.messages,
            "model": MODEL_ALIASES.get(request.model_engine, request.model_engine),
            "logprobs": 0,
        }

    @staticmethod
    def convert_to_raw_completion_request(request: Request) -> Dict:
        # Following the examples from https://github.com/togethercomputer/open-models-api
        return {
            "temperature": request.temperature,
            "n": request.num_completions,
            "max_tokens": request.max_tokens,
            "stop": request.stop_sequences or None,
            "echo": request.echo_prompt,
            "top_p": request.top_p,
            "top_k": request.top_k_per_token,
            "prompt": request.prompt,
            "model": MODEL_ALIASES.get(request.model_engine, request.model_engine),
            "logprobs": 0,
        }

    def __init__(self, cache_config: CacheConfig, api_key: Optional[str] = None):
        # When an API key is not specified in credentials.conf, 
        # we rely on offline evaluation only.
        self.api_key: Optional[str] = api_key
        self.cache = Cache(cache_config)
        self.model_type = None

    def set_model_type(self, model_type):
        self.model_type = model_type

    def make_request(self, request: Request) -> RequestResult:

        if self.model_type == "CHAT":
            raw_request = TogetherAIClient.convert_to_raw_chat_request(request)
        else:
            raw_request = TogetherAIClient.convert_to_raw_completion_request(request)
        cache_key: Dict = Client.make_cache_key(raw_request, request)

        if not self.api_key:
            raise TogetherAIClientError("togetherApiKey not set in credentials.conf")
        
        headers: Dict[str, str] = {"Authorization": f"Bearer {self.api_key}"}

        def do_it_sync() -> Dict[Any, Any]:
            raw_response = None

            if self.model_type == "CHAT":
                raw_response = requests.post(TogetherAIClient.CHAT_ENDPOINT, headers=headers, json=raw_request)
            else:
                raw_response = requests.post(TogetherAIClient.COMPLETION_ENDPOINT, headers=headers, json=raw_request)

            try:
                raw_response.raise_for_status()
            except Exception as e:
                raise TogetherAIClientError(
                    f"Together request failed with {raw_response.status_code}: {raw_response.text}"
                ) from e
            
            # result = response.json()
            # if "output" not in result:
            #     raise TogetherAIClientError(f"Could not get output from Together response: {result}")
            # if "error" in result["output"]:
            #     error_message = result["output"]["error"]
            #     raise TogetherAIClientError(f"Together request failed with error: {error_message}")
            # return result["output"]

            return raw_response.json()

        try:
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it_sync))
        except Exception as error:
            return RequestResult(
                success=False,
                cached=False,
                error=str(error),
                completions=[],
                embedding=[],
            )

        # Expect the result to be structured the same way as a response from OpenAI API.
        completions: List[Sequence] = []
        for raw_completion in response.get("choices", []):
            sequence_logprob = 0
            tokens: List[Token] = []

            # Currently, token_logprobs is provided in interactive/online mode but it has a different format
            # Waiting for a fix.
            if raw_completion.get("logprobs", None):
                raw_data = raw_completion.get("logprobs")
                for text, logprob, top_logprobs in zip(
                    raw_data.get("tokens"), raw_data.get("token_logprobs"), raw_data.get("top_logprobs")
                ):
                    text = cleanup_str(text, "together")
                    tokens.append(Token(text=text, logprob=logprob or 0, top_logprobs=dict(top_logprobs or {})))
                    sequence_logprob += logprob or 0
            else:
                # hack: just make the entire text one token so that something shows up in the frontend
                if self.model_type == "CHAT":
                    text = cleanup_str(raw_completion.get("message").get("content"), "together")
                else:
                    text = cleanup_str(raw_completion.get("text"), "together")
                tokens.append(Token(text=text, logprob=0, top_logprobs={}))

            raw_finish_reason: Optional[str] = raw_completion.get("finish_reason")
            finish_reason: Optional[Dict] = {"reason": raw_finish_reason} if raw_finish_reason else None

            if self.model_type == "CHAT":
                completion = Sequence(
                    text=cleanup_str(raw_completion.get("message").get("content"), "together"),
                    logprob=sequence_logprob,
                    tokens=tokens,
                    finish_reason=finish_reason,
                )
            else:
                completion = Sequence(
                    text=cleanup_str(raw_completion.get("text"), "together"),
                    logprob=sequence_logprob,
                    tokens=tokens,
                    finish_reason=finish_reason,
                )
            completion = truncate_sequence(completion, request)
            completions.append(completion)

        request_time: Union[float, Dict[str, Any]] = response.get("request_time", 0.0)
        if isinstance(request_time, dict):
            batch_performance_metadata: Dict = response.get("request_time")
            return RequestResult(
                success=True,
                cached=cached,
                request_time=0,
                completions=completions,
                batch_size=batch_performance_metadata.get("batch_size"),
                batch_request_time=batch_performance_metadata.get("batch_time"),
                embedding=[],
                raw_response=response
            )
        else:
            return RequestResult(
                success=True,
                cached=cached,
                request_time=response["raw_compute_time"] if "raw_compute_time" in response else request_time,
                completions=completions,
                embedding=[],
                raw_response=response
            )

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        raise NotImplementedError("Use the HuggingFaceClient to tokenize.")

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        raise NotImplementedError("Use the HuggingFaceClient to decode.")

