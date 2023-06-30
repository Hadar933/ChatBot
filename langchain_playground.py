import os

import openai
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain

# %%
with open('API_KEYS') as f:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = f.readlines()[0]
    os.environ["OPENAI_API_TOKEN"] = f.readlines()[1]
# %%

repo_id = "google/flan-t5-xl"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options

llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0, "max_length": 64})
# %%
template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "Who won the FIFA World Cup in the year 1994? "

print(llm_chain.run(question))
