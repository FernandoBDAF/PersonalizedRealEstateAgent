# Chroma:

## Search types you can do with Chroma (via LangChain)
### Similarity search (text query)
vectorstore.similarity_search(query: str, k=4, filter: dict|None=None) -> list[Document]

### Similarity search with scores
vectorstore.similarity_search_with_score(query: str, k=4, filter: dict|None=None) -> list[tuple[Document, float]]
### Max Marginal Relevance (diverse results)
vectorstore.max_marginal_relevance_search(query: str, k=4, fetch_k=20, lambda_mult=0.5, filter: dict|None=None) -> list[Document]
### Search by precomputed embedding vector
vectorstore.similarity_search_by_vector(embedding: list[float], k=4, filter: dict|None=None)
vectorstore.max_marginal_relevance_search_by_vector(embedding: list[float], ...)

Notes:
Use filter={...} to constrain by metadata (e.g., {"neighborhood": "Sunnyvale"}).
To get a vector: embedding = OpenAIEmbeddings(...).embed_query(query_text).