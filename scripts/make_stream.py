"""Generate stream.jsonl (30 deduplicated questions across 5 thematic clusters) and planted.json (5 false beliefs)."""

import json
from collections import defaultdict
from pathlib import Path

raw_path = Path("data/raw/hotpot_dev_distractor_v1.json")
raw_data = json.loads(raw_path.read_text(encoding="utf-8"))

# Frozen IDs to avoid overlap with debug, dev, test
frozen_ids = set()
for split in ["debug", "dev", "test"]:
    sp_path = Path(f"data/manifests/{split}.jsonl")
    if sp_path.exists():
        for line in sp_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                frozen_ids.add(json.loads(line)["_id"])

# Map titles to available items
t_map = defaultdict(list)
for q in raw_data:
    if q["_id"] not in frozen_ids:
        for t, _ in q["supporting_facts"]:
            t_map[t].append(q)

target_clusters = [
    ("Music Composers", ["Paul Hindemith", "Maurice Ravel", "Béla Bartók", "Bernhard Heiden", "Benjamin Britten", "Suvini Zerboni"]),
    ("WWII Battles", ["Battle of Manila (1945)", "Battle of Guam (1944)", "Battle of Saipan", "Battle of Stones River", "Spring 1945 offensive in Italy", "Seven Days Battles"]),
    ("Universities", ["University of Colorado", "Columbia University", "University of Rochester", "Ateneo de Davao University", "University of Leeds", "University of Vermont"]),
    ("Film Directors", ["Wolfgang Becker", "Paul Andrew Williams", "Vladimir Danilevich", "Jacques Audiard", "Jean Vigo", "Lester James Peries", "The Lone Ranger (2013 film)", "Poetry in Motion (film)"]),
    ("Actors & Writers", ["Elizabeth Taylor", "Margaret Landon", "Antonin Artaud", "Henry Roth", "Robert Erskine Childers", "Terry Richardson", "Li Yitong (actress)", "Unbroken (film)"]),
]

selected_stream = []
seen_qids = set()
seen_questions = set()

for cluster_name, titles in target_clusters:
    cluster_items = []
    # Collect candidate questions matching any title in this cluster
    candidates = []
    for t in titles:
        for q in t_map.get(t, []):
            if q["_id"] not in seen_qids and q["question"].strip().lower() not in seen_questions:
                candidates.append(q)
    
    # Pick 6 distinct questions for this cluster
    for q in candidates:
        if q["_id"] not in seen_qids and q["question"].strip().lower() not in seen_questions:
            seen_qids.add(q["_id"])
            seen_questions.add(q["question"].strip().lower())
            cluster_items.append(q)
            if len(cluster_items) == 6:
                break
    
    print(f"Cluster '{cluster_name}': selected {len(cluster_items)} questions.")
    selected_stream.extend(cluster_items)

assert len(selected_stream) == 30, f"Expected 30 questions, got {len(selected_stream)}"

# Save stream.jsonl
stream_path = Path("data/manifests/stream.jsonl")
with open(stream_path, "w", encoding="utf-8") as f:
    for q in selected_stream:
        f.write(json.dumps(q) + "\n")
print(f"Saved 30 stream questions to {stream_path}")

# 5 Planted wrong beliefs for override test E5
planted_beliefs = [
    {
        "text": "Paul Hindemith was born in 1875 in Frankfurt, Germany.",
        "conf": 0.95,
        "target_entity": "Paul Hindemith",
        "description": "Falsely asserts Hindemith was born in 1875 (actual 1895).",
    },
    {
        "text": "The Battle of Manila occurred in January 1943 during the Pacific War.",
        "conf": 0.95,
        "target_entity": "Battle of Manila (1945)",
        "description": "Falsely asserts Battle of Manila occurred in 1943 (actual 1945).",
    },
    {
        "text": "University of Colorado is a private research university founded by John Evans.",
        "conf": 0.95,
        "target_entity": "University of Colorado",
        "description": "Falsely asserts University of Colorado is private (actual public).",
    },
    {
        "text": "Wolfgang Becker is a French film director born in Paris.",
        "conf": 0.95,
        "target_entity": "Wolfgang Becker",
        "description": "Falsely asserts Wolfgang Becker is French (actual German).",
    },
    {
        "text": "Elizabeth Taylor was born on February 27, 1890 in London.",
        "conf": 0.95,
        "target_entity": "Elizabeth Taylor",
        "description": "Falsely asserts Elizabeth Taylor was born in 1890 (actual 1932).",
    },
]

planted_path = Path("data/manifests/planted.json")
with open(planted_path, "w", encoding="utf-8") as f:
    json.dump(planted_beliefs, f, indent=2)
print(f"Saved 5 planted wrong beliefs to {planted_path}")
