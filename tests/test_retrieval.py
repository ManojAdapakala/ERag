import json
import numpy as np
from src.retrieval.index import Embedder, Retriever, RetrieverFactory, BM25Retriever, paragraphs_from_item


def test_embedder_normalization_and_memo():
    emb = Embedder("sentence-transformers/all-MiniLM-L6-v2")
    texts = ["Albert Einstein was a theoretical physicist.", "Quantum mechanics is fundamental."]
    vecs = emb.encode(texts)

    assert vecs.shape[0] == 2
    assert vecs.shape[1] == 384  # MiniLM dimension
    # Test L2 normalization (norm should be 1.0)
    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-5)

    # Test memoization
    assert texts[0] in emb.memo
    assert texts[1] in emb.memo


def test_retriever_search_ranking():
    emb = Embedder("sentence-transformers/all-MiniLM-L6-v2")
    paras = [
        {"title": "Doc A", "sentences": ["The capital of France is Paris."], "text": "Doc A. The capital of France is Paris."},
        {"title": "Doc B", "sentences": ["The capital of Germany is Berlin."], "text": "Doc B. The capital of Germany is Berlin."},
        {"title": "Doc C", "sentences": ["Apples are edible fruits."], "text": "Doc C. Apples are edible fruits."},
    ]
    retriever = Retriever(paras, emb)
    results = retriever.search("What is the French capital?", k=2)

    assert len(results) == 2
    assert results[0]["title"] == "Doc A"
    assert results[0]["score"] > results[1]["score"]


def test_distractor_retriever_factory():
    emb = Embedder("sentence-transformers/all-MiniLM-L6-v2")
    sample_item = {
        "_id": "test_1",
        "question": "Where is the Eiffel Tower?",
        "context": [
            ["Paris", ["The Eiffel Tower is a wrought-iron lattice tower located in Paris."]],
            ["London", ["The Big Ben is located in London."]],
        ],
        "supporting_facts": [["Paris", 0]],
    }
    factory = RetrieverFactory(setting="distractor", embedder=emb)
    r = factory.for_item(sample_item)
    results = r.search(sample_item["question"], k=1)
    assert len(results) == 1
    assert results[0]["title"] == "Paris"


def test_manifest_integrity_and_disjointness():
    with open("data/manifests/manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["debug"]["n"] == 50
    assert manifest["dev"]["n"] == 100
    assert manifest["test"]["n"] == 400

    debug_ids = {json.loads(l)["_id"] for l in open("data/manifests/debug.jsonl", "r", encoding="utf-8")}
    dev_ids = {json.loads(l)["_id"] for l in open("data/manifests/dev.jsonl", "r", encoding="utf-8")}
    test_ids = {json.loads(l)["_id"] for l in open("data/manifests/test.jsonl", "r", encoding="utf-8")}

    assert len(debug_ids) == 50
    assert len(dev_ids) == 100
    assert len(test_ids) == 400

    # Ensure zero overlap
    assert len(debug_ids & dev_ids) == 0
    assert len(debug_ids & test_ids) == 0
    assert len(dev_ids & test_ids) == 0


def test_gold_supporting_facts_present_in_distractor_context():
    # Verify for all 50 debug questions, the gold supporting facts exist in context
    with open("data/manifests/debug.jsonl", "r", encoding="utf-8") as f:
        debug_items = [json.loads(l) for l in f]

    for item in debug_items:
        context_titles = {t for t, _ in item["context"]}
        gold_titles = {t for t, _ in item["supporting_facts"]}
        assert gold_titles.issubset(context_titles), f"Gold titles missing in context for question {item['_id']}"


def test_bm25_retriever_ranking_and_factory():
    paras = [
        {"title": "Doc A", "sentences": ["Albert Einstein was a theoretical physicist.", "He won the Nobel Prize."], "text": "Doc A. Albert Einstein was a theoretical physicist. He won the Nobel Prize."},
        {"title": "Doc B", "sentences": ["Marie Curie was a pioneering physicist and chemist.", "She won two Nobel Prizes."], "text": "Doc B. Marie Curie was a pioneering physicist and chemist. She won two Nobel Prizes."},
        {"title": "Doc C", "sentences": ["The Eiffel Tower is located in Paris, France."], "text": "Doc C. The Eiffel Tower is located in Paris, France."},
    ]
    bm25 = BM25Retriever(paras)
    res = bm25.search("Who was Albert Einstein the physicist?", k=1)
    assert len(res) == 1
    assert res[0]["title"] == "Doc A"
    assert res[0]["score"] > 0.0

    # Test via RetrieverFactory
    sample_item = {
        "_id": "bm25_test",
        "question": "Where is the Eiffel Tower?",
        "context": [
            ["Doc A", ["Albert Einstein was a physicist."]],
            ["Doc C", ["The Eiffel Tower is located in Paris, France."]],
        ],
    }
    factory = RetrieverFactory(setting="distractor", retriever_type="bm25")
    ret = factory.for_item(sample_item)
    res_factory = ret.search("Eiffel Tower Paris", k=1)
    assert len(res_factory) == 1
    assert res_factory[0]["title"] == "Doc C"

