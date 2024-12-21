import time
import asyncio
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.schema import Document
from llama_index.core.storage.docstore.simple_docstore import SimpleDocumentStore
import numpy as np
from typing import Literal
from functools import lru_cache
import tiktoken

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

async def run_performance_test():
    # Create test document and store
    docstore = SimpleDocumentStore()

    with open('prideandprejudice.txt', 'r') as f:
        large_text = f.read()
        
    doc = Document(text=large_text)
    docstore.add_documents([doc])
    
    # Test each operation separately
    n_runs = 5
    token_count_times = []
    truncate_times = []
    docstore_times = []
    
    for _ in range(n_runs):
        # Test token counting
        start = time.time()
        _ = count_tokens(large_text)
        token_count_times.append(time.time() - start)
        
        # Test truncation
        start = time.time()
        _ = truncate_text(large_text, 1000)
        truncate_times.append(time.time() - start)
        
        # Test document retrieval
        start = time.time()
        _ = await docstore.aget_document(doc.doc_id)
        docstore_times.append(time.time() - start)
    
    print("\nToken counting:")
    print(f"Average: {np.mean(token_count_times):.3f}s")
    print(f"Min: {min(token_count_times):.3f}s")
    print(f"Max: {max(token_count_times):.3f}s")
    
    print("\nTruncation:")
    print(f"Average: {np.mean(truncate_times):.3f}s")
    print(f"Min: {min(truncate_times):.3f}s")
    print(f"Max: {max(truncate_times):.3f}s")
    
    print("\nDocument retrieval:")
    print(f"Average: {np.mean(docstore_times):.3f}s")
    print(f"Min: {min(docstore_times):.3f}s")
    print(f"Max: {max(docstore_times):.3f}s")

if __name__ == "__main__":
    asyncio.run(run_performance_test())