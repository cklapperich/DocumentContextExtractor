from llama_index.core.llms import ChatMessage
from typing import Dict, List
from llama_index.core.llms.llm import LLM
from llama_index.core.async_utils import DEFAULT_NUM_WORKERS, run_jobs
from llama_index.core.extractors import BaseExtractor
from llama_index.core.schema import NodeRelationship
from llama_index.core import Settings
from textwrap import dedent
import importlib
import logging
from llama_index.core.storage.docstore.simple_docstore import DocumentStore

DEFAULT_CONTEXT_PROMPT: str = dedent("""
    Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Full sentences not neccessary. Disambiguate pronouns and key terms in the chunk. Answer only with the succinct context and nothing else.
    """).strip()

DEFAULT_KEY: str = "context"
from llama_index.core.node_parser import TokenTextSplitter

class DocumentContextExtractor(BaseExtractor):
    keys: List[str]
    prompts: List[str]
    docstore: DocumentStore
    llm: LLM
    max_context_length: int
    max_contextual_tokens: int
    oversized_document_strategy: str


    @staticmethod
    def _truncate_text(text: str, max_token_count: int, how='first') -> str:
        """
        Truncate text to the specified token count
        :param text: The text to truncate
        :param max_token_count: The maximum number of tokens to return
        :param how: How to truncate the text. Can be 'first' or 'last'
        :return: The truncated text
        """
        text_splitter = TokenTextSplitter(chunk_size=max_token_count, chunk_overlap=0)
        chunks = text_splitter.split_text(text)
        if how == 'first':
            text = chunks[0]
        elif how == 'last':
            text = chunks[-1]
        else:
            raise ValueError("Invalid truncation method. Must be 'first' or 'last'.")
        
        return text if text else ""
    
    @staticmethod
    def _count_tokens(text: str) -> int:
        """Count tokens in text"""
        text_splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)
        tokens = text_splitter.split_text(text)
        return len(tokens)
            
    def __init__(
        self,
        docstore: DocumentStore,
        llm: LLM = None,
        keys=None,
        prompts=None,
        num_workers: int = DEFAULT_NUM_WORKERS,
        max_context_length: int = 128000,
        max_contextual_tokens: int = 512,
        oversized_document_strategy: str = "truncate_first",
        **kwargs
    ):
        """Initialize the extractor.
        
        Args:
            llm (LLM): LLM to use for context extraction. Mandatory.
            docstore: DocumentStore to use for context extraction. Mandatory.
            keys (List[str]): List of keys to extract context for
            prompts (List[str]): List of prompts to use for context extraction 

            num_workers (int): Number of workers to use for context extraction
            max_context_length (int): Maximum context length to use for context extraction
            max_contextual_tokens (int): Maximum contextual tokens to use for context extraction
            oversized_document_strategy (str): Strategy to use for documents > max_context_length:
                "truncate_first" - Truncate from top down
                "truncate_last" - Truncate from bottom up
                "warn" - Warn about oversized document
                "error" - Raise error for oversized document
                "ignore" - Skip oversized document
        """
        if not importlib.util.find_spec("tiktoken"):
            raise ValueError("TikToken is required for DocumentContextExtractor. Please install tiktoken.")

        # Process defaults
        keys = keys or [DEFAULT_KEY]
        prompts = prompts or [DEFAULT_CONTEXT_PROMPT]

        if isinstance(keys, str):
            keys = [keys]
        if isinstance(prompts, str):
            prompts = [prompts]
    
        llm = llm or Settings.llm

        # Call super().__init__ with processed values
        super().__init__(
            keys=keys,
            prompts=prompts,
            llm=llm,
            docstore=docstore,
            num_workers=num_workers,
            max_context_length=max_context_length,
            oversized_document_strategy=oversized_document_strategy,
            max_contextual_tokens=max_contextual_tokens,
            **kwargs
        )

    async def _agenerate_node_context(self, node, metadata, document_content, prompt, key)->Dict:
 
        """Generate context for a node using parent document content."""
        cached_text = f"<document>{document_content}</document>"

        messages = [
            ChatMessage(
                role="user",
                content=[
                    {
                        "text": cached_text,
                        "block_type": "text",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "text": f"Here is the chunk we want to situate within the whole document:\n<chunk>{node.get_content()}</chunk>\n{prompt}",
                        "block_type": "text",
                    },
                ],
            ),
        ]
        response = await self.llm.achat(
            messages,
            max_tokens=self.max_contextual_tokens,
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        )
        metadata[key] = response.message.content
        return metadata
    
    async def aextract(self, nodes) -> List[Dict]:
        """Extract context metadata for nodes using parent relationships."""
        # Initialize metadata list preserving node order
        metadata_list = [{} for _ in nodes]
        node_jobs = []

        # we need to preserve the order of the nodes, but process the nodes uot-of-order
        metadata_map = {node.node_id: metadata_dict for metadata_dict, node in zip(metadata_list, nodes)}
        source_doc_ids = set([node.source_node.node_id for node in nodes])

        # make a mapping of doc id: node
        doc_id_to_nodes = {}
        for node in nodes:
            if not (node.source_node and (node.source_node.node_id in source_doc_ids)):
                continue
            parent_id = node.source_node.node_id
            
            if parent_id not in doc_id_to_nodes:
                doc_id_to_nodes[parent_id] = []
            doc_id_to_nodes[parent_id].append(node)

        i = 0
        for doc_id in source_doc_ids:
            doc = self.docstore.get_document(doc_id)
            doc_content = doc.text
            # Handle oversized parent content
            if self.max_context_length is not None:
                token_count = self._count_tokens(doc_content)
                if token_count > self.max_context_length:
                    message = f"Document {doc.id} is too large ({token_count} tokens) to be processed. Doc metadata: {doc.metadata}"
                    
                    if self.oversized_document_strategy == "truncate_first":
                        doc_content = self._truncate_text(doc_content, self.max_context_length, how='first')
                    elif self.oversized_document_strategy == "truncate_last":
                        doc_content = self._truncate_text(doc_content, self.max_context_length, how='last')
                    elif self.oversized_document_strategy == "warn":
                        logging.warning(message)
                    elif self.oversized_document_strategy == "error":
                        raise ValueError(message)
                    elif self.oversized_document_strategy == "ignore":
                        continue
                    else:
                        raise ValueError(f"Unknown oversized document strategy: {self.oversized_document_strategy}")

            # Queue up context generation for each prompt
            node_summaries_jobs = []
            for prompt, key in list(zip(self.prompts, self.keys)):
                for node in doc_id_to_nodes.get(doc_id,[]):
                    i += 1
                    metadata_dict = metadata_map[node.node_id]
                    node_summaries_jobs.append(self._agenerate_node_context(node, metadata_dict, doc_content, prompt, key))

        # Run all jobs in parallel if we have any
        if node_jobs:
            results = await run_jobs(
                [job[1] for job in node_jobs],
                show_progress=self.show_progress,
                workers=self.num_workers,
            )
            
            # Update metadata with results
            for (i, _), result in zip(node_jobs, results):
                metadata_list[i].update(result)

        return metadata_list