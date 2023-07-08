import os
import pickle
import re

import requests
import dotenv
from langchain import HuggingFaceHub
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredURLLoader

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from xml.etree import ElementTree
from typing import Optional, List, Pattern
from loguru import logger

dotenv.load_dotenv(os.path.abspath('API_KEYS.env'))


class AIFactory:
    @staticmethod
    def embedding_factory(platform: Optional[str] = 'huggingface'):
        match platform:
            case 'openai':
                return OpenAIEmbeddings()
            case 'huggingface':
                return HuggingFaceEmbeddings()
            case _:
                return None

    @staticmethod
    def model_factory(repo_id: Optional[str] = 'tiiuae/falcon-7b-instruct'):
        match repo_id:
            case 'openai':
                return ChatOpenAI()
            case other:
                return HuggingFaceHub(repo_id=repo_id,
                                      model_kwargs={"max_new_tokens": 500})


class Crawler:
    """
    The Crawler class parses a site's URL and extract all redirected sub-URLs within it
    """

    def __init__(self, site_url: str, url_pattern: Optional[Pattern[str]] = None):
        self.site_url: str = site_url
        self.sitemap_url: str = f"{site_url}//sitemap.xml"
        self.url_pattern: Pattern[str] = re.compile(url_pattern) if url_pattern is not None else None
        self.urls: List[str] = []

    def _apply_regex_filter(self) -> List[str]:
        """ filters the url based on a desired pattern (regex) """
        sub_urls = [u for u in self.urls if re.search(self.url_pattern, u)]
        if len(sub_urls) == 0:
            logger.warning(f"None of the URLs matched with {self.url_pattern}. Using all URLs instead.")
            return self.urls
        return sub_urls

    def extract_urls(self) -> List[str]:
        """
         Extracts all URLs from a sitemap XML string.
        """
        # Parse the XML from the string
        schema = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
        all_sitemaps = requests.get(self.sitemap_url).text
        root = ElementTree.fromstring(all_sitemaps)
        # extracting all possible sub-sitemaps (in case the sitemap_url is an index)
        sub_sitemaps = root.findall(f'{schema}sitemap')
        if len(sub_sitemaps) == 0:  # the sitemap does not redirect to sub-sitemaps
            sub_sitemaps = [self.sitemap_url]
        else:
            sub_sitemaps = [xml.find(f'{schema}loc').text for xml in sub_sitemaps]
        urls = []
        for xml in sub_sitemaps:
            xml_instance = ElementTree.fromstring(requests.get(xml).text)
            for url_object in xml_instance.findall(f'{schema}url'):
                curr_url = url_object.find(f'{schema}loc').text
                urls.append(curr_url)
        if self.url_pattern:
            urls = self._apply_regex_filter()
        return urls


class ChatBot:
    """
    The ChatBot class takes in a site url and converts in to
    a vector database from which one can extract information based on
    vector similarity to a query's input
    """

    def __init__(self, site_url: str,
                 llm, embedding,
                 chunk_size: Optional[int] = 2000,
                 chunk_overlap: Optional[int] = 500,
                 url_pattern: Optional[str] = None):
        self.llm = llm
        self.embedding = embedding
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_db_path = os.path.join('vector_db_cache', 'vdb.pkl')

        self.crawler = Crawler(site_url, url_pattern)
        self.vector_db = self.build_vdb()
        self.chain = self.build_chain()

    def build_chain(self):
        """ returns a RetrievalQAWithSources chain given a specified llm, type and database """
        logger.info(f"Building RetrievalQAWithSourcesChain...")
        return RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.llm,
            chain_type="map_reduce",
            retriever=self.vector_db.as_retriever()
        )

    def build_vdb(self):
        """ returns a vector database, either from memory or from scratch """
        if os.path.exists(self.vector_db_path):  # fast load
            logger.info(f"Loading Vector-Database from {self.vector_db_path}...")
            with open(self.vector_db_path, "rb") as file:
                return pickle.load(file)
        else:
            logger.info(f"Building Vector-Database.")
            urls = self.crawler.extract_urls()
            chunks = self._from_urls_to_chunks(urls)
            vector_db = FAISS.from_documents(chunks, self.embedding)
            with open(self.vector_db_path, "wb") as file:
                pickle.dump(vector_db, file)
            return vector_db

    def _from_urls_to_chunks(self, urls):
        """ takes the list of urls and converts in to a dataset of chunks with given size and overlap """
        loader = UnstructuredURLLoader(urls)
        data = loader.load()
        doc_splitter = CharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = doc_splitter.split_documents(data)
        return chunks

    def query(self, prompt: str):
        result = self.chain({"question": prompt}, return_only_outputs=True)
        pretty_print(result)
        return result


def pretty_print(dictionary, line_length: Optional[int] = 70):
    """ prints out the result of our chatbot """
    answer = dictionary['answer']
    sources = dictionary['sources'].split(',')
    answer_to_print = '\n'.join([answer[i:i + line_length] for i in range(0, len(answer), line_length)])
    print("Answer:")
    print(answer_to_print, end='\r')
    print("-" * line_length)
    print("Sources:", end='\r')
    for i, src in enumerate(sources):
        print(f"{src}")


if __name__ == "__main__":
    language_model = AIFactory.model_factory('openai')
    embedding_model = AIFactory.embedding_factory('openai')
    url = 'https://ayalatours.co.il/'

    db = ChatBot(site_url=url, llm=language_model, embedding=embedding_model)

    # Ask a question
    prompt = "what is this website about?"
    res = db.query(prompt)
    x = 2
