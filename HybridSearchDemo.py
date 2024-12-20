from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.storage.index_store.simple_index_store import SimpleIndexStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, AsyncQdrantClient
from llama_index.embeddings.nvidia import NVIDIAEmbedding

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openrouter import OpenRouter
from llama_index.core.postprocessor import LLMRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.storage.docstore.simple_docstore import SimpleDocumentStore
    
import os
from src.DocumentContextExtractor import DocumentContextExtractor
class HybridSearchWithContext:
    CHUNK_SIZE = 228
    CHUNK_OVERLAP = 50
    SIMILARITY_TOP_K = 10
    SPARSE_TOP_K = 20
    REREANKER_TOP_N = 3
    def __init__(self, name:str):
        """
        :param name: The name of the index, required for the underlying vector store
        """
        # Initialize clients
        client = QdrantClient(":memory:")
        aclient = AsyncQdrantClient(":memory:")
        self.index_store_path = f"{name}"

        if not os.path.exists(self.index_store_path):
            os.makedirs(self.index_store_path)

        # Load documents
        self.context_llm = OpenRouter(model="openai/gpt-4o-mini")
        self.answering_llm = OpenRouter(model="openai/gpt-4o-mini")

        self.embed_model = NVIDIAEmbedding("nvidia/nv-embedqa-mistral-7b-v2", embed_batch_size=20)

        sample_embedding = self.embed_model.get_query_embedding("sample text")
        self.embed_size = len(sample_embedding)

        self.reranker = LLMRerank(
                    choice_batch_size=5,
                    top_n=self.REREANKER_TOP_N,
                    llm=self.context_llm
                )

        # Create vector store
        self.vector_store = QdrantVectorStore(
            name,
            client=client,
            aclient=aclient,
            enable_hybrid=True,
            batch_size=20,
            dim=self.embed_size
        )

        # Initialize storage context
        if os.path.exists(os.path.join(self.index_store_path, "index_store.json")):
            index_store=SimpleIndexStore.from_persist_dir(persist_dir=self.index_store_path)
        else:
            index_store=SimpleIndexStore()

        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store,
                                                            index_store=index_store)

        # Create text splitter
        self.text_splitter = SentenceSplitter(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP
        )

        # DocumentContextExtractor requires a document store
        self.document_context_extractor = DocumentContextExtractor(docstore=self.storage_context.docstore, llm=self.context_llm, max_context_length=128000,
                                                                   max_contextual_tokens=228, oversized_document_strategy="truncate_first")

        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embed_model,
            storage_context=self.storage_context,   
            transformations=[self.text_splitter, self.document_context_extractor]
        )

        self.storage_context.persist(persist_dir=self.index_store_path)
    
    def add_directory(self, directory):
        reader = SimpleDirectoryReader(directory)
        documents = reader.load_data()

        self.storage_context.docstore.add_documents(documents)
        for doc in documents:
            self.index.insert(doc)

        self.query_engine = self.index.as_query_engine(
            similarity_top_k=self.SIMILARITY_TOP_K, 
            sparse_top_k=self.SPARSE_TOP_K, 
            vector_store_query_mode="hybrid",
            llm=self.answering_llm,
            node_postprocessors=[self.reranker]
        )

        self.retriever = self.index.as_retriever(
            similarity_top_k=self.SIMILARITY_TOP_K,
            sparse_top_k=self.SPARSE_TOP_K,
            vector_store_query_mode="hybrid"
        )

        self.storage_context.persist(persist_dir=self.index_store_path)

    def get_raw_search_results(self, question):

        retrieved_nodes = self.retriever.retrieve(question)
        
        retrieved_texts = [node.text for node in retrieved_nodes]
        
        return retrieved_nodes
    
    def query_engine(self, question):
        response = self.query_engine.query(
            question
        )

        return response
    
if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv()
    hybrid_search = HybridSearchWithContext(name="hybriddemo")
    hybrid_search.add_directory("./data")

    question = "Why was this document written?"
    print(hybrid_search.get_raw_search_results(question))