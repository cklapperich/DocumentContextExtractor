from llama_index.core.llms import ChatMessage, LLM
from llama_index.core.async_utils import DEFAULT_NUM_WORKERS, run_jobs
from llama_index.core.extractors import BaseExtractor
from llama_index.core.schema import Document, Node
from llama_index.core import Settings
from llama_index.core.storage.docstore.simple_docstore import DocumentStore
from llama_index.core.node_parser import TokenTextSplitter
from typing import Optional, Dict, List, Tuple, Set, Union, Literal, Any
from textwrap import dedent
import importlib
import logging
import asyncio
import random
from functools import lru_cache
import tiktoken

OversizeStrategy = Literal["truncate_first", "truncate_last", "warn", "error", "ignore"]
MetadataDict = Dict[str, str]

DEFAULT_CONTEXT_PROMPT: str = dedent("""
Generate keywords and brief phrases describing the main topics, entities, and actions in this text. 
Replace pronouns with their specific referents. Format as comma-separated phrases. 
Exclude meta-commentary about the text itself.
""").strip()

DEFAULT_KEY: str = "context"

class DocumentContextExtractor(BaseExtractor):
    """
    Extracts contextual information from documents chunks using LLM-based analysis for enhanced RAG accuracy. Based on the Anthropic "Contextual Retrieval" blog post. 
    This extractor processes documents and their nodes to generate contextual metadata,
    handling rate limits and large documents according to specified strategies.
    """
    
    # Pydantic fields
    llm: Any
    docstore: DocumentStore
    key: str
    prompt: str
    doc_ids: Set[str]
    max_context_length: int
    max_contextual_tokens: int
    oversized_document_strategy: OversizeStrategy
    warn_on_oversize: bool = True
    tiktoken_encoder: str

    def __init__(
        self,
        docstore: DocumentStore,
        llm: LLM,
        key: Optional[str] = DEFAULT_KEY,
        prompt: Optional[str] = DEFAULT_CONTEXT_PROMPT,
        num_workers: int = DEFAULT_NUM_WORKERS,
        max_context_length: int = 128000,
        max_contextual_tokens: int = 512,
        oversized_document_strategy: OversizeStrategy = "truncate_first",
        warn_on_oversize: bool = True,
        tiktoken_encoder: str = "cl100k_base",
        **kwargs
    ) -> None:
        """
        Initialize the DocumentContextExtractor.
        
        Args:
            docstore: DocumentStore llama_index object, database for storing the parent documents of the incoming nodes.
            keys: Key(s) for storing extracted context
            prompts: Prompt(s) for context extraction
            llm: Language model for generating context
            num_workers: Number of parallel workers
            max_context_length: Maximum document context length
            max_contextual_tokens: Maximum tokens in generated context
            oversized_document_strategy: How to handle documents exceeding max_context_length
            **kwargs: Additional parameters for BaseExtractor
            
        Raises:
            ValueError: If tiktoken is not installed or if invalid strategy is provided
        """
        if not importlib.util.find_spec("tiktoken"):
            raise ValueError("TikToken is required for DocumentContextExtractor. Please install tiktoken.")

        # Process input parameters
     
        llm = llm or Settings.llm
        doc_ids: Set[str] = set()

        super().__init__(
            key=key,
            prompt=prompt,
            llm=llm,
            docstore=docstore,
            num_workers=num_workers,
            doc_ids=doc_ids,
            max_context_length=max_context_length,
            oversized_document_strategy=oversized_document_strategy,
            max_contextual_tokens=max_contextual_tokens,
            warn_on_oversize=warn_on_oversize,
            tiktoken_encoder=tiktoken_encoder,
            **kwargs
        )

    @staticmethod
    @lru_cache(maxsize=1000)
    def _count_tokens(text: str, encoder:str="cl100k_base") -> int:
        encoding = tiktoken.get_encoding(encoder)
        return len(encoding.encode(text))

    @staticmethod
    def _truncate_text(text: str, max_token_count: int, how: Literal['first', 'last'] = 'first', encoder="cl100k_base") -> str:
        encoding = tiktoken.get_encoding(encoder)
        tokens = encoding.encode(text)
        
        if how == 'first':
            truncated_tokens = tokens[:max_token_count]
        else:  # last
            truncated_tokens = tokens[-max_token_count:]
            
        return encoding.decode(truncated_tokens)

    async def _agenerate_node_context(
        self,
        node: Node,
        metadata: MetadataDict,
        document: Document,
        prompt: str,
        key: str
    ) -> MetadataDict:
        """
        Generate context for a node using LLM with retry logic.
        Implements exponential backoff for rate limit handling. Uses prompt caching when available.
        
        Args:
            node: Node to generate context for
            metadata: Metadata dictionary to update
            document: Parent document containing node
            prompt: Prompt for context generation
            key: Key for storing generated context
            
        Returns:
            Updated metadata dictionary

        """
        cached_text = f"<document>{document.text}</document>"
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
                        "text": f"Here is the chunk we want to situate within the whole document:\n<chunk>{node.text}</chunk>\n{prompt}",
                        "block_type": "text",
                    },
                ],
            ),
        ]

        max_retries = 5
        base_delay = 60

        for attempt in range(max_retries):
            try:
                response = await self.llm.achat(
                    messages,
                    max_tokens=self.max_contextual_tokens,
                    extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
                )
                metadata[key] = response.message.blocks[0].text
                return metadata

            except Exception as e:
                is_rate_limit = any(
                    message in str(e).lower()
                    for message in ["rate limit", "too many requests", "429"]
                )

                if is_rate_limit and attempt < max_retries - 1:
                    delay = (base_delay * (2 ** attempt)) + (random.random() * 0.5)
                    logging.warning(
                        f"Rate limit hit, retrying in {delay:.1f} seconds "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                
                if is_rate_limit:
                    logging.error(f"Failed after {max_retries} retries due to rate limiting")
                else:
                    logging.warning(f"Error generating context for node {node.node_id}: {str(e)}")
                
                return metadata
            
    async def aextract(self, nodes: List[Node]) -> List[MetadataDict]:
        """
        Extract context for multiple nodes asynchronously, optimized for loosely ordered nodes.
        Processes each node independently without guaranteeing sequential document handling.
        Includes LRU caching for document fetching.
        
        Args:
            nodes: List of nodes to process, ideally grouped by source document
            
        Returns:
            List of metadata dictionaries with generated context
        """
        metadata_list = [{} for _ in nodes]
        metadata_map = {node.node_id: metadata_dict for metadata_dict, node in zip(metadata_list, nodes)}        


        async def _get_document(doc_id: str) -> Document:
            """counting tokens can be slow, as can awaiting the docstore (potentially), so we keep a small lru_cache"""

            # first we need to get the document
            doc = await self.docstore.aget_document(doc_id)

            # then truncate if necessary. 
            if self.max_context_length is not None:
                strategy = self.oversized_document_strategy
                token_count = self._count_tokens(doc.text, self.tiktoken_encoder)
                if token_count > self.max_context_length:
                    message = (
                        f"Document {doc.id} is too large ({token_count} tokens) "
                        f"to be processed. Doc metadata: {doc.metadata}"
                    )
                    
                    if self.warn_on_oversize:
                        logging.warning(message)

                    if strategy == "truncate_first":
                        doc.text = self._truncate_text(doc.text, self.max_context_length, 'first', self.tiktoken_encoder)
                    elif strategy == "truncate_last":
                        doc.text = self._truncate_text(doc.text, self.max_context_length, 'last', self.tiktoken_encoder)                    
                    elif strategy == "error":
                        raise ValueError(message)
                    elif strategy == "ignore":
                        return
                    else:
                        raise ValueError(f"Unknown oversized document strategy: {strategy}")
                
            return doc

        # iterate over all the nodes and generate the jobs
        node_tasks = []
        for node in nodes:
            if not node.source_node:
                return

            doc = await _get_document(node.source_node.node_id)

            if not doc:
                continue

            metadata = metadata_map[node.node_id]
            task = self._agenerate_node_context(node, metadata, doc, self.prompt, self.key)
            node_tasks.append(task)
        
        # then run the jobs
        await run_jobs(
            node_tasks,
            show_progress=self.show_progress,
            workers=self.num_workers,
        )
        
        return metadata_list