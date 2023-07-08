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

    def __init__(self, site_url: str, url_contains: Optional[str] = None):
        self.site_url: str = site_url
        self.sitemap_url: str = f"{site_url}//sitemap.xml"
        self.url_contains: Pattern[str] = re.compile(url_contains) if url_contains is not None else None
        self.schema = "{http://www.sitemaps.org/schemas/sitemap/0.9}"

    def filter_urls(self, urls: List[str]):
        """ filters the url based on a desired patter
        """
        sub_urls = [u for u in urls if re.search(self.url_contains, u)]
        return urls if len(sub_urls) == 0 else sub_urls

    def extract_urls(self):
        """
         Extract all URLs from a sitemap XML string.
        :return:  A list of URLs extracted from the sitemap.
        """
        # Parse the XML from the string
        logger.info(f"Loading sitemap from {self.site_url} ...")
        all_sitemaps = requests.get(self.sitemap_url).text
        root = ElementTree.fromstring(all_sitemaps)
        # extracting all possible sub-sitemaps (in case the sitemap_url is an index)
        sub_sitemaps = root.findall(f'{self.schema}sitemap')
        if len(sub_sitemaps) == 0:  # the sitemap does not redirect to sub-sitemaps
            sub_sitemaps = [self.sitemap_url]
        else:
            sub_sitemaps = [xml.find(f'{self.schema}loc').text for xml in sub_sitemaps]
        urls = []
        for xml in sub_sitemaps:
            xml_instance = ElementTree.fromstring(requests.get(xml).text)
            for url_object in xml_instance.findall(f'{self.schema}url'):
                curr_url = url_object.find(f'{self.schema}loc').text
                urls.append(curr_url)
        return urls


class VectorDB:
    """
    The VectorDB class takes in a site url and converts in to
    a vector database from which one can extract information based on
    vector similarity to a query's input
    """

    def __init__(self, site_url: str,
                 llm, embedding,
                 chunk_size: Optional[int] = 2000,
                 chunk_overlap: Optional[int] = 500,
                 url_contains: Optional[str] = None):
        self.llm = llm
        self.embedding = embedding
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_db_path = os.path.join('vector_db_cache', 'vdb.pkl')
        self.crawler = Crawler(site_url, url_contains)
        if os.path.exists(self.vector_db_path):  # fast load
            self.vector_db = self._get_vector_db(from_memory=True)
        else:  # slow load
            self.urls = self.crawler.extract_urls()
            if url_contains:
                self.urls = self.crawler.filter_urls(self.urls)
            self.chunks = self._from_urls_to_chunks()
            self.vector_db = self._get_vector_db()
        self.chain = self._build_chain()

    def _build_chain(self):
        """ returns a RetrievalQAWithSources chain given a specified llm, type and database """
        logger.info("Building the retrieval chain ...")
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.llm,
            chain_type="map_reduce",
            retriever=self.vector_db.as_retriever()
        )
        return chain

    def _get_vector_db(self, from_memory: bool = False):
        """
        returns a chroma vector database, either from memory or from scratch
        :param from_memory: True iff the vector db was previously saved to memory
        """
        if from_memory:
            logger.info(f"Loading {self.vector_db_path} from memory.")
            with open(self.vector_db_path, "rb") as file:
                return pickle.load(file)
        else:
            logger.info("Building the vector database ...")
            vector_db = FAISS.from_documents(self.chunks, self.embedding)
            logger.info(f"Saving vector db to memory as as {self.vector_db_path}.")
            with open(self.vector_db_path, "wb") as file:
                pickle.dump(vector_db, file)
            return vector_db

    def _from_urls_to_chunks(self):
        """
        takes the list of urls and converts in to a dataset of chunks with given size and overlap
        :return: chunks data structure
        """
        logger.info("Loading URLs content ...")
        loader = UnstructuredURLLoader(self.urls)
        data = loader.load()
        logger.info("Splitting documents in chunks ...")
        doc_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = doc_splitter.split_documents(data)
        logger.info(f"{len(chunks)} chunks created")
        return chunks

    def query(self, prompt: str):
        return self.chain({"question": prompt}, return_only_outputs=True)


if __name__ == "__main__":
    language_model = AIFactory.model_factory('openai')
    embedding_model = AIFactory.embedding_factory('openai')
    url = 'https://ayalatours.co.il/'

    db = VectorDB(site_url=url, llm=language_model, embedding=embedding_model, url_contains='days')

    # Ask a question
    prompt = "can you summarize all the weekly picks in this website?"
    res = db.query(prompt)
    x = 2
