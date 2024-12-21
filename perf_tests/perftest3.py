import time
import tiktoken
from functools import lru_cache
import numpy as np

@lru_cache(maxsize=1000)
def count_tokens_cached(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def count_tokens_direct(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def run_test():
    # Create test documents of different sizes
    sizes = [1000, 10000, 100000]
    
    for size in sizes:
        text = f"This is a test document. " * size
        print(f"\nTesting with document size {size} words:")
        
        # Test uncached first run
        start = time.time()
        _ = count_tokens_cached(text)
        first_cache_time = time.time() - start
        
        # Test cached second run
        start = time.time()
        _ = count_tokens_cached(text)
        second_cache_time = time.time() - start
        
        # Test direct tokenization
        start = time.time()
        _ = count_tokens_direct(text)
        direct_time = time.time() - start
        
        print(f"First run with cache: {first_cache_time:.3f}s")
        print(f"Second run with cache: {second_cache_time:.3f}s")
        print(f"Direct tokenization: {direct_time:.3f}s")

if __name__ == "__main__":
    run_test()