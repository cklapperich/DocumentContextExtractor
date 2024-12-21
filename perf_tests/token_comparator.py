import tiktoken

def compare_encodings(text: str):
    encodings = ["cl100k_base", "p50k_base", "r50k_base"]
    results = {}
    
    for enc_name in encodings:
        encoding = tiktoken.get_encoding(enc_name)
        count = len(encoding.encode(text))
        results[enc_name] = count
    
    return results

# Test with different types of text
test_cases = {
    "Simple English": "This is a simple test of the encoding systems.",
    "Technical": "The DocumentContextExtractor class implements efficient token-based text processing with configurable chunking strategies.",
    "Mixed": "Here's some code: for i in range(10): print(f'Value: {i}')",
    "Special Chars": "Special characters like 你好, üñîçødé, and emojis 🌟 can affect encoding.",
    "Long Technical": """The transformer architecture employs multi-headed self-attention mechanisms to process sequential data. 
    Each attention head can learn different aspects of the relationships between tokens in the sequence. 
    The model uses positional encodings to maintain sequence order information.""" * 3
}

for name, text in test_cases.items():
    print(f"\n{name}:")
    counts = compare_encodings(text)
    base = counts['cl100k_base']
    for enc, count in counts.items():
        diff = ((count - base) / base) * 100
        print(f"{enc}: {count} tokens ({diff:+.1f}% vs cl100k)")