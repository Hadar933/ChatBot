import os
import openai
from typing import Optional, Dict

import yaml
from langchain import HuggingFaceHub
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate


def load_api_keys(key: Optional[str] = None,
                  val: Optional[str] = None) -> None:
    """
    Either load a specific key or the default OpenAI and HuggingFace Keys from the `API_KEYS` file
    :param key: key name for the os. environ
    :param val: the key itself
    """
    if key:
        os.environ[key] = str(val)
    else:
        with open('API_KEYS.yaml', 'r') as f:
            data = yaml.safe_load(f)
            for key, val in data.items():
                os.environ[key] = str(val)


def load_huggingface_llm(repo_id: Optional[str] = "google/flan-t5-xl", model_kwargs: Dict = None):
    """
    loads a hugging-face model
    :param repo_id: See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for more options
    :param model_kwargs: a dictionary with possible model arguments
    :return: hf llm model object
    """
    model_kwargs = model_kwargs if model_kwargs is not None else {"temperature": 0, "max_length": 64}
    hf_llm = HuggingFaceHub(repo_id=repo_id, model_kwargs=model_kwargs)
    return hf_llm


def load_openai_llm(temperature: Optional[float] = 0.0):
    """
    loads an OpenAI model (that could be in Chat)
    :param temperature: value within [0,1] for results randomization
    :return: openai model object
    """

    openai_llm = OpenAI(openai_api_key=os.environ['OPENAI_API_KEY'], temperature=temperature)
    return openai_llm


def generate_QA_prompt():
    question_prompt = PromptTemplate.from_template(
        "Question: {question}\n"
        "Let's think step by step."
    )
    return question_prompt


def answer_question(llm, prompt_template, question):
    chain = LLMChain(llm=llm, prompt=prompt_template)
    ans = chain.run(question)
    return ans


if __name__ == '__main__':
    load_api_keys()
    llm = load_openai_llm()
    template = generate_QA_prompt()
    question = "Can you provide an itinerary for Phi Phi island, Thailand?"
    answer = answer_question(llm, template, question)
    print(answer)
