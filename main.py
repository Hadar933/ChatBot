import os
from typing import Optional

from langchain import HuggingFaceHub
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

import requests
from xml.etree import ElementTree
from loguru import logger

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '...'
os.environ['OPENAI_API_KEY'] = '...'


def ai_factory(platform: str):
    """
    a factory for different models
    :param platform: a string that represents the desired platform
    :return: a dictionary with embedding model and chat model
    """
    match platform:
        case 'openai':
            return {
                'embedding': OpenAIEmbeddings(),
                'model': ChatOpenAI()
            }

        case 'huggingface':
            return {
                'embedding': HuggingFaceEmbeddings(),
                'model': HuggingFaceHub(
                    repo_id="google/flan-t5-xl",
                    model_kwargs={"temperature": 0, "max_length": 64}
                )

            }
        case other:
            raise ValueError(f"Unsupported platform {platform}.")


class VectorDB:
    """
    The VectorDB class takes in a site url and converts in to
    a vector database from which one can extract information based on
    vector similarity to a query's input
    """

    def __init__(self, site_url: str,
                 chunk_size: Optional[int] = 8000,
                 chunk_overlap: Optional[int] = 3000,
                 url_contains: Optional[str] = None,
                 platform: Optional[str] = 'openai'):
        self.site_url = site_url
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.pattern = url_contains
        self.sitemap_url = f"{url}//sitemap.xml"
        self.platform = platform

        self.ai = ai_factory(platform)
        self.urls = self._extract_urls_from_sitemap()
        if self.pattern:
            self.urls = self._filter_urls()
        self.chunks = self._from_urls_to_chunks()
        self.chain = self._build_chain()

    def _build_chain(self):
        logger.info("Building the vector database ...")
        docsearch = Chroma.from_documents(self.chunks, self.ai['embedding'])
        logger.info("Building the retrieval chain ...")
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.ai['model'],
            chain_type="map_reduce",
            retriever=docsearch.as_retriever(),
        )
        return chain

    def _from_urls_to_chunks(self):
        """
        takes the list of urls and converts in to a dataset of chunks with given size and overlap
        :param chunk_overlap: over-lap between consecutive chunks of data
        :param chunk_size: number of tokens per chunk
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

    def _filter_urls(self):
        """
        filters the url based on a desired patter
        """
        logger.info(f"Filtering URLs with pattern {self.pattern} ...")
        sub_urls = [x for x in self.urls if self.pattern in x]
        if len(sub_urls) == 0:
            logger.info(f"No matching urls for {self.pattern}. Using ALL Urls instead.")
            return self.urls
        else:
            logger.info(f"{len(sub_urls)} URLs extracted")
            return sub_urls

    def _extract_urls_from_sitemap(self):
        """
         Extract all URLs from a sitemap XML string.
        :return:  A list of URLs extracted from the sitemap.
        """
        # Parse the XML from the string
        logger.info(f"Loading sitemap from {self.site_url} ...")
        sitemaps = requests.get(self.sitemap_url).text
        root = ElementTree.fromstring(sitemaps)
        # Define the namespace for the sitemap XML
        namespace = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        # Find all <loc> elements under the <url> elements
        urls = []
        for url_object in root.findall("ns:url", namespace):
            curr_url = url_object.find("ns:loc", namespace).text
            urls.append(curr_url)
        return urls

    def query(self, prompt: str):

        return self.chain({"question": prompt}, return_only_outputs=True)


if __name__ == "__main__":
    # Build the knowledge base
    url = 'https://ayalatours.co.il/'
    db = VectorDB(
        site_url=url,
        url_contains='/Israel/',
        platform='huggingface'
    )

    # Ask a question
    res = db.query("Are there any deals to Karpathops?")
