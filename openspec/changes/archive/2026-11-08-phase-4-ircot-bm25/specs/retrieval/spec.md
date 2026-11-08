# Change Specification Delta: BM25 Lexical Retrieval

## ADDED Requirements

### Requirement: BM25 Lexical Paragraph Retrieval
The system SHALL provide an in-memory sparse lexical BM25 retriever conforming to the retriever interface.

#### Scenario: Lexical query retrieval
- **GIVEN** a collection of candidate paragraphs
- **WHEN** a text query is submitted to `BM25Retriever.search(query, k)`
- **THEN** it returns up to $k$ paragraphs ranked by BM25 scoring with non-negative scores.

#### Scenario: Retriever factory selection
- **GIVEN** a question item and `retriever_type="bm25"`
- **WHEN** `RetrieverFactory.for_item(item)` is invoked
- **THEN** it constructs and returns an initialized `BM25Retriever` instance for that item's context.
