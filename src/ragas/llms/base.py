from __future__ import annotations

import asyncio
import logging
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
import json
from typing import Any, Coroutine

from langchain_community.chat_models import ChatVertexAI
from langchain_community.llms import VertexAI
from langchain_core.callbacks import Callbacks
from langchain_core.language_models import BaseLanguageModel
from langchain_core.outputs import LLMResult
from langchain_openai.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain_openai.llms import AzureOpenAI, OpenAI
from langchain_openai.llms.base import BaseOpenAI
from ragas.llms.prompt import PromptValue

from ragas.run_config import RunConfig, add_async_retry, add_retry
from ragas.utils import api_inference, format_prompt
from .prompt_tllm import AdvanceInstructSample, json_grammar


if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.llms.prompt import PromptValue

logger = logging.getLogger(__name__)

MULTIPLE_COMPLETION_SUPPORTED = [
    OpenAI,
    ChatOpenAI,
    AzureOpenAI,
    AzureChatOpenAI,
    ChatVertexAI,
    VertexAI,
]


def is_multiple_completion_supported(llm: BaseLanguageModel) -> bool:
    """Return whether the given LLM supports n-completion."""
    for llm_type in MULTIPLE_COMPLETION_SUPPORTED:
        if isinstance(llm, llm_type):
            return True
    return False


@dataclass
class BaseRagasLLM(ABC):
    run_config: RunConfig

    def set_run_config(self, run_config: RunConfig):
        self.run_config = run_config

    def get_temperature(self, n: int) -> float:
        """Return the temperature to use for completion based on n."""
        return 0.3 if n > 1 else 1e-8

    @abstractmethod
    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 1e-8,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        ...

    @abstractmethod
    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 1e-8,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        ...

    async def generate(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 1e-8,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
        is_async: bool = True,
    ) -> LLMResult:
        """Generate text using the given event loop."""
        if is_async:
            agenerate_text_with_retry = add_async_retry(
                self.agenerate_text, self.run_config
            )
            return await agenerate_text_with_retry(
                prompt=prompt,
                n=n,
                temperature=temperature,
                stop=stop,
                callbacks=callbacks,
            )
        else:
            loop = asyncio.get_event_loop()
            generate_text_with_retry = add_retry(self.generate_text, self.run_config)
            generate_text = partial(
                generate_text_with_retry,
                prompt=prompt,
                n=n,
                temperature=temperature,
                stop=stop,
                callbacks=callbacks,
            )
            return await loop.run_in_executor(None, generate_text)


class LangchainLLMWrapper(BaseRagasLLM):
    """
    A simple base class for RagasLLMs that is based on Langchain's BaseLanguageModel
    interface. it implements 2 functions:
    - generate_text: for generating text from a given PromptValue
    - agenerate_text: for generating text from a given PromptValue asynchronously
    """

    def __init__(
        self, langchain_llm: BaseLanguageModel, run_config: t.Optional[RunConfig] = None
    ):
        self.langchain_llm = langchain_llm
        if run_config is None:
            run_config = RunConfig()
        self.set_run_config(run_config)

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 1e-8,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        temperature = self.get_temperature(n=n)
        if is_multiple_completion_supported(self.langchain_llm):
            return self.langchain_llm.generate_prompt(
                prompts=[prompt],
                n=n,
                temperature=temperature,
                stop=stop,
                callbacks=callbacks,
            )
        else:
            result = self.langchain_llm.generate_prompt(
                prompts=[prompt] * n,
                temperature=temperature,
                stop=stop,
                callbacks=callbacks,
            )
            # make LLMResult.generation appear as if it was n_completions
            # note that LLMResult.runs is still a list that represents each run
            generations = [[g[0] for g in result.generations]]
            result.generations = generations
            return result

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 1e-8,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        temperature = self.get_temperature(n=n)
        if is_multiple_completion_supported(self.langchain_llm):
            return await self.langchain_llm.agenerate_prompt(
                prompts=[prompt],
                n=n,
                temperature=temperature,
                stop=stop,
                callbacks=callbacks,
            )
        else:
            result = await self.langchain_llm.agenerate_prompt(
                prompts=[prompt] * n,
                temperature=temperature,
                stop=stop,
                callbacks=callbacks,
            )
            # make LLMResult.generation appear as if it was n_completions
            # note that LLMResult.runs is still a list that represents each run
            generations = [[g[0] for g in result.generations]]
            result.generations = generations
            return result

    def set_run_config(self, run_config: RunConfig):
        self.run_config = run_config

        # configure if using OpenAI API
        if isinstance(self.langchain_llm, BaseOpenAI) or isinstance(
            self.langchain_llm, ChatOpenAI
        ):
            try:
                from openai import RateLimitError
            except ImportError:
                raise ImportError(
                    "openai.error.RateLimitError not found. Please install openai package as `pip install openai`"
                )
            self.langchain_llm.request_timeout = run_config.timeout
            self.run_config.exception_types = RateLimitError


def llm_factory(
    model: str = "gpt-3.5-turbo-16k", run_config: t.Optional[RunConfig] = None
) -> BaseRagasLLM:
    timeout = None
    if run_config is not None:
        timeout = run_config.timeout
    openai_model = ChatOpenAI(model=model, timeout=timeout)
    return LangchainLLMWrapper(openai_model, run_config)


class EndpointModel(BaseRagasLLM):
    def __init__(self, url, run_config: t.Optional[RunConfig] = None):
        self.url = url + "/completion"
        self.headers = {
            "Content-Type": "application/json"
        }
        if run_config is None:
            run_config = RunConfig()
        self.set_run_config(run_config)
        

    def generate_text(self, 
        text: str, 
        temperature=0.7, 
        dynatemp_range = 0.3 ,
        n_keep = -1 ,
        grammar = "",
        top_p = 0.45,
        min_p = 0.045, 
        seed = -1,
        top_k = 60,
        repeat_penalty = 1.15,
        presence_penalty = 0,
        frequency_penalty = 0,
        new_session_stop_word = "[NEW]",
        stream = True,
        cache_prompt = True,
        system_prompt=None,
        **kwargs
    ):
        grammar = json_grammar if grammar == "json_grammar" else ""
        

        default_system_prompt = """
        You're an AI Large Language Model developed(created) by an AI developer named Tuấn Phạm, your task are to think loudly step by step before give a good and relevant response
        to the user request based on their provided documents, answer in the language the user preferred. Only using the provided knowledge, not using your pretrained knowledge.

        The following is a conversation with an AI Large Language Model. The AI has been trained to answer questions, provide recommendations, and help with decision making.
        The AI follows user requests. The AI thinks outside the box.

        The AI will take turn in a multi-turn dialogs conversation with the user, stay in context with the previous chat.
        """
        qas_id = "TEST"
        orig_answer_texts = "TEST"

        prompted_input = system_prompt or default_system_prompt

        final_message = prompted_input + f"Base on the provided documents, answer the following question:\n" + text

        config_prompt = format_prompt(
            AdvanceInstructSample, 
            {"qas_id": qas_id,
            "system_prompt": prompted_input,
            "orig_answer_texts": orig_answer_texts,
            "question_text": final_message
        })
        

        data = {
            "prompt": config_prompt,
            "temperature": temperature,
            "dynatemp_range": dynatemp_range,
            "n_keep": n_keep,
            "stream": stream,
            "cache_prompt": cache_prompt,
            "grammar": grammar,
            "top_p": top_p,
            "min_p": min_p,
            "seed": seed,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        }

        client = api_inference(self.url, self.headers, data, verify=False, stream=True)
        assistant_message = ''
        for event in client.events():
            payload = json.loads(event.data)
            chunk = payload['content']
            assistant_message += chunk
        fake_output = LLMResult(
            generations=[[assistant_message]],
        )
        return fake_output
    
    async def agenerate_text(self, prompt: PromptValue, n: int = 1, temperature: float = 1e-8, stop: List[str] | None = None, callbacks: Callbacks = None) -> LLMResult:
        ...