# Spec Delta: Retrieval

## Purpose
Provides unified multi-retriever interfaces across lexical and dense architectures, supporting asymmetric prefixing and rigorous comparative evaluation across retrieval models.

## ADDED Requirements

### Requirement: Multi-Model Dense and Lexical Retrieval Support
The retrieval subsystem SHALL support instantiating and executing lexical BM25 (`bm25`) and multiple dense bi-encoder models (`sentence-transformers/all-MiniLM-L6-v2`, `intfloat/e5-base-v2`, `BAAI/bge-base-en-v1.5`) via a standardized retriever interface.

#### Scenario: Instantiation of 768-dimensional dense model
- **WHEN** the caller requests a retriever with model name `intfloat/e5-base-v2` or `BAAI/bge-base-en-v1.5`
- **THEN** the retrieval subsystem instantiates a dense retriever that computes 768-dimensional normalized embeddings and performs inner product similarity scoring.

#### Scenario: Instantiation of lexical retriever
- **WHEN** the caller requests a retriever with type `bm25`
- **THEN** the retrieval subsystem instantiates a BM25 retriever utilizing `BM25Plus` formulation with non-negative scoring.

### Requirement: Asymmetric Query and Passage Formatting
The dense retrieval subsystem SHALL apply model-specific asymmetric prefixes to queries and passages when required by the underlying embedding model.

#### Scenario: E5 query and passage prefixing
- **WHEN** encoding documents and queries with `intfloat/e5-base-v2`
- **THEN** the system prepends `passage: ` to every passage before encoding, and prepends `query: ` to every search query before encoding.

#### Scenario: BGE query instruction prefixing
- **WHEN** querying with `BAAI/bge-base-en-v1.5`
- **THEN** the system prepends `Represent this sentence for searching relevant passages: ` to the query without modifying document passage representations.

### Requirement: Test-400 Comparative Matrix Generation
The evaluation subsystem SHALL evaluate Standard RAG, IRCoT, and E-RAG-Balanced across all four retrieval models on the frozen HotpotQA Test-400 dataset and record a 12-system comparative matrix.

#### Scenario: Evaluation artifact generation
- **WHEN** the multi-retriever benchmark completes execution
- **THEN** the system writes out a structured comparative CSV containing Exact Match, Token F1, Gold Recall, All-Gold Rate, Selective EM, ECE, AUROC, Calls, Tokens, and Cost across all 12 experimental conditions.
