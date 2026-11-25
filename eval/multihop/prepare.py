"""Build a multi-hop retrieval benchmark from HippoRAG's released dev sets (MuSiQue, 2WikiMultiHopQA).

Data: https://github.com/OSU-NLP-Group/HippoRAG/tree/legacy/data (downloaded into eval/multihop/data/).
Each sampled question brings its own candidate passages (supporting + distractors), as HippoRAG built its corpus.
Gold = the question's supporting passages; one passage = one LightRAG document whose file_source is the passage key.
Bridge = the intermediate entities of the reasoning chain (what PPR is meant to lift):
  MuSiQue: answers of every sub-question but the last; 2Wiki: entities that are the object of one evidence
  triple and the subject of another (e.g. Lothair II -mother-> Ermengarde of Tours -died-> 851).

Usage:
  python eval/multihop/prepare.py --dataset musique --n 300 --seed 0
  python eval/multihop/prepare.py --dataset 2wiki --n 300            # bridge types only (compositional, inference)
Output: eval/multihop/<dataset>_n<N>/{questions.json, corpus.json}
"""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
FILES = {"musique": "musique.json", "2wiki": "2wikimultihopqa.json"}
WIKI_BRIDGE_TYPES = {"compositional", "inference"}  # comparison types do not need a bridge entity


def key(title, text):
    return "mh-" + hashlib.sha1(f"{title}\n{text}".encode()).hexdigest()[:12]


def musique(x):
    paras = [(p["title"], p["paragraph_text"], p["is_supporting"]) for p in x["paragraphs"]]
    bridge = [d["answer"] for d in x["question_decomposition"][:-1]]
    return x["id"], int(x["id"][0]), paras, bridge


def wiki(x):
    gold_titles = {t for t, _ in x["supporting_facts"]}
    paras = [(t, " ".join(s), t in gold_titles) for t, s in x["context"]]
    subj, obj = {e[0] for e in x["evidences"]}, {e[2] for e in x["evidences"]}
    return x["_id"], 2, paras, sorted(subj & obj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=FILES, default="musique")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    raw = json.load(open(HERE / "data" / FILES[a.dataset], encoding="utf-8"))
    if a.dataset == "2wiki":
        raw = [x for x in raw if x["type"] in WIKI_BRIDGE_TYPES]
    sample = random.Random(a.seed).sample(raw, min(a.n, len(raw)))

    questions, corpus = [], {}
    for x in sample:
        qid, hops, paras, bridge = (musique if a.dataset == "musique" else wiki)(x)
        gold = []
        for title, text, sup in paras:
            k = key(title, text)
            corpus[k] = {"key": k, "title": title, "text": text}
            if sup:
                gold.append(k)
        questions.append({"id": qid, "question": x["question"], "answer": x["answer"], "hops": hops, "gold": gold, "bridge": bridge})

    out = HERE / f"{a.dataset}_n{len(questions)}"
    out.mkdir(exist_ok=True)
    json.dump(questions, open(out / "questions.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(list(corpus.values()), open(out / "corpus.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{out}: {len(questions)} questions {dict(Counter(q['hops'] for q in questions))}, {len(corpus)} passages")


if __name__ == "__main__":
    main()
