from llama_index.core.llms import ChatMessage, LLM
from llama_index.core.async_utils import DEFAULT_NUM_WORKERS, run_jobs
from llama_index.core.extractors import BaseExtractor
from llama_index.core.schema import Document, Node
from llama_index.core import Settings
from llama_index.core.storage.docstore.simple_docstore import DocumentStore
from llama_index.core.node_parser import TokenTextSplitter
from typing import Optional, Dict, List, Tuple, Set, Union, Literal
from textwrap import dedent
import importlib
import logging
import asyncio
import random

# Type definitions
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
    Extracts contextual information from documents using LLM-based analysis.
    
    This extractor processes documents and their nodes to generate contextual metadata,
    handling rate limits and large documents according to specified strategies.
    
    Attributes:
        keys (List[str]): Keys used to store extracted context in metadata
        prompts (List[str]): Prompts used for context extraction
        llm (LLM): Language model instance for generating context
        docstore (DocumentStore): Storage for document data
        doc_ids (Set[str]): Set of processed document IDs
        max_context_length (int): Maximum allowed context length in tokens
        max_contextual_tokens (int): Maximum tokens for generated context
        oversized_document_strategy (OversizeStrategy): Strategy for handling large documents
    """
    
    # Pydantic fields
    llm: LLM
    docstore: DocumentStore
    keys: List[str]
    prompts: List[str]
    doc_ids: Set[str]
    max_context_length: int
    max_contextual_tokens: int
    oversized_document_strategy: OversizeStrategy

    def __init__(
        self,
        docstore: DocumentStore,
        llm: LLM,
        keys: Optional[Union[str, List[str]]] = None,
        prompts: Optional[Union[str, List[str]]] = None,
        num_workers: int = DEFAULT_NUM_WORKERS,
        max_context_length: int = 128000,
        max_contextual_tokens: int = 512,
        oversized_document_strategy: OversizeStrategy = "truncate_first",
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
        keys = [keys] if isinstance(keys, str) else (keys or [DEFAULT_KEY])
        prompts = [prompts] if isinstance(prompts, str) else (prompts or [DEFAULT_CONTEXT_PROMPT])
        llm = llm or Settings.llm
        doc_ids: Set[str] = set()

        super().__init__(
            keys=keys,
            prompts=prompts,
            llm=llm,
            docstore=docstore,
            num_workers=num_workers,
            doc_ids=doc_ids,
            max_context_length=max_context_length,
            oversized_document_strategy=oversized_document_strategy,
            max_contextual_tokens=max_contextual_tokens,
            **kwargs
        )

    @staticmethod
    def _truncate_text(
        text: str,
        max_token_count: int,
        how: Literal['first', 'last'] = 'first'
    ) -> str:
        """
        Truncate text to specified token count from either start or end.
        
        Args:
            text: Text to truncate
            max_token_count: Maximum number of tokens to keep
            how: Whether to keep first or last portion
            
        Returns:
            Truncated text string
            
        Raises:
            ValueError: If invalid truncation method specified
        """
        text_splitter = TokenTextSplitter(chunk_size=max_token_count, chunk_overlap=0)
        chunks = text_splitter.split_text(text)
        
        if not chunks:
            return ""
            
        if how == 'first':
            return chunks[0]
        elif how == 'last':
            return chunks[-1]
            
        raise ValueError("Invalid truncation method. Must be 'first' or 'last'.")

    @staticmethod
    def _count_tokens(text: str) -> int:
        """
        Count tokens in text using TokenTextSplitter.
        
        Args:
            text: Text to count tokens in
            
        Returns:
            Number of tokens in text
        """
        text_splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)
        tokens = text_splitter.split_text(text)
        return len(tokens)

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
        
        Args:
            node: Node to generate context for
            metadata: Metadata dictionary to update
            document: Parent document containing node
            prompt: Prompt for context generation
            key: Key for storing generated context
            
        Returns:
            Updated metadata dictionary
            
        Note:
            Implements exponential backoff for rate limit handling
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
        Extract context for multiple nodes asynchronously.
        
        Args:
            nodes: List of nodes to process
            
        Returns:
            List of metadata dictionaries with generated context
            
        Note:
            Handles document size limits according to oversized_document_strategy
        """
        metadata_list = [{} for _ in nodes]
        metadata_map = {node.node_id: metadata_dict for metadata_dict, node in zip(metadata_list, nodes)}

        source_doc_ids = {node.source_node.node_id for node in nodes if node.source_node}
        doc_id_to_nodes: Dict[str, List[Node]] = {}

        for node in nodes:
            if not (node.source_node and node.source_node.node_id in source_doc_ids):
                continue
            parent_id = node.source_node.node_id
            doc_id_to_nodes.setdefault(parent_id, []).append(node)

        for doc_id in source_doc_ids:
            doc = self.docstore.get_document(doc_id)

            if self.max_context_length is not None:
                token_count = self._count_tokens(doc.text)
                if token_count > self.max_context_length:
                    message = (
                        f"Document {doc.id} is too large ({token_count} tokens) "
                        f"to be processed. Doc metadata: {doc.metadata}"
                    )

                    if self.oversized_document_strategy == "truncate_first":
                        doc.text = self._truncate_text(doc.text, self.max_context_length, 'first')
                    elif self.oversized_document_strategy == "truncate_last":
                        doc.text = self._truncate_text(doc.text, self.max_context_length, 'last')
                    elif self.oversized_document_strategy == "warn":
                        logging.warning(message)
                    elif self.oversized_document_strategy == "error":
                        raise ValueError(message)
                    elif self.oversized_document_strategy == "ignore":
                        continue
                    else:
                        raise ValueError(f"Unknown oversized document strategy: {self.oversized_document_strategy}")

            node_summaries_jobs = [
                self._agenerate_node_context(node, metadata_map[node.node_id], doc, prompt, key)
                for prompt, key in zip(self.prompts, self.keys)
                for node in doc_id_to_nodes.get(doc_id, [])
            ]

            await run_jobs(
                node_summaries_jobs,
                show_progress=self.show_progress,
                workers=self.num_workers,
            )
            logging.info(f"Processed {len(node_summaries_jobs)} nodes with {self.num_workers} workers.")

        return metadata_list