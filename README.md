
# DEPRECATED. THIS HAS NOW BEEN MERGED INTO LLAMA_INDEX. PLEASE USE LLAMA_INDEX.CORE.EXTRACTORS.DocumentContextExtractor https://github.com/run-llama/llama_index/pull/17367

## Summary

This repository contains a llama_index implementation of "contextual retrieval" (https://www.anthropic.com/news/contextual-retrieval)

It implements a custom llama_index Extractor class, which can then be used in a llama index pipeline. It requires you to initialize it using a Document Store and an LLM to provide the context. It also requires you keep the documentstore up to date. 

## motivation

Anthropic made a 'cookbook' notebook to demo this. llama_index also made a demo of it here: https://docs.llamaindex.ai/en/stable/examples/cookbooks/contextual_retrieval

The problem, there are tons of edge cases when trying to replicate what this at scale, over 100s of documents: 

- rate limits are a huge problem

- cost

- I want to put this into a pipeline

- documents too large for context window

- prompt caching doesn't work via llama_index interface

- error handling

- chunk + context can be too big for the embedding model

- and much more!

## Demo

See hybridsearchdemo.py for a demo of the extractor in action with Qdrant hybrid search, effectively re-implementing the blog post. All the OTHER parts of the blog post (reranking, hybrid search) are already well implemented in llama_index, in my opinion.

## Usage

```python
docstore = SimpleDocumentStore()

llm = OpenRouter(model="openai/gpt-4o-mini")

# initialize the extractor
extractor = DocumentContextExtractor(document_store, llm)

storage_context = StorageContext.from_defaults(vector_store=self.vector_store,
                                                            docstore=docstore,
                                                            index_store=index_store)
index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
            storage_context=storage_context,
            transformations=[text_splitter, self.document_context_extractor]
        )

reader = SimpleDirectoryReader(directory)
documents = reader.load_data()

# have to keep this updated for the DocumentContextExtractor to function.
storagecontext.docstore.add_documents(documents)
for doc in documents:
    self.index.insert(doc)
```

### custom keys and prompts

by default, the extractor adds a key called "context" to each node, using a reasonable default prompt taken from the blog post cookbook, but you can pass in a list of keys and prompts like so:

```python
extractor = DocumentContextExtractor(document_store, llm, keys=["context", "title"], prompts=["Give the document context", "Provide a chunk title"])
```

## model selection

You need something fast, high rate limits, long context, low cost on input tokens.

Recommended models:

- gpt 4o-mini

- Gemini flash models

- long-context local models (would love recommendations)

gpt 4o-mini is king. The 128k context, smart, automatic prompt caching make it absolutely perfect. Throw $50 at openai and wait 7 days, they'll give you 2mil tokens/minute at $0.075/mil toks.
You're going to pay (doc_size * doc_size//chunk_size) tokens for each document in input costs, and then (num_chunks * 200) or so for output tokens.
This means 10-50 million tokens to process Pride and Prejudice, if you dont split it into chapters first.


## TODO
- TEST 'succinct' prompt performance vs 'full' prompt performance!
- Support for Batch requests (supported by anthropic and openai) to handle truly massive amounts of documents?
- fix this bug because it prevents Llama index from working with Python 3.10: https://github.com/run-llama/llama_index/discussions/14351
- add a TransformComponent that splits documents into smaller documents and then adds them to the docstore
    - or better yet, a TransformComponent that simply adds the nodes to the docstore and does nothing else
    - then you can build a pipeline like this: ChapterSplitter -> DocstoreCatcher -> SentenceSplitter -> DocumentContextExtractor
- make a pull request to llama_index
