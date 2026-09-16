"""Deterministic synthetic benchmark corpus (nonpolitical everyday topics).

    python benchmarks/corpus.py --out benchmarks/corpus-v1.json

Texts combine topic clauses with sentiment and sentence forms, so tasks have a mix of
clear and ambiguous items. Lengths: short (one sentence), medium (4-6 sentences),
long (30-45 sentences, several hundred tokens).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random

CORPUS_VERSION = 1

TOPICS = {
    "weather": ["the rain started just after lunch", "the wind picked up near the coast", "the morning fog lasted until noon",
                "it snowed lightly overnight", "the afternoon was unusually warm", "a thunderstorm rolled through the valley"],
    "food": ["the soup at the corner cafe", "the bread from the new bakery", "our dinner at the noodle place",
             "the strawberries from the market", "the coffee in the office kitchen", "the lemon cake my neighbor baked"],
    "travel": ["the train to the coast", "our flight home", "the ferry across the bay",
               "the hotel near the station", "the bus tour of the old town", "the long drive through the mountains"],
    "sports": ["the local football match", "my morning swim", "the cycling race on Sunday",
               "the tennis lesson", "the school basketball game", "the neighborhood fun run"],
    "technology": ["the new phone update", "my laptop battery", "the printer at the library",
                   "the video call software", "the smart thermostat", "the photo editing app"],
    "health": ["my visit to the dentist", "the new stretching routine", "the flu shot appointment",
               "my sleep over the past week", "the physiotherapy session", "the walk after dinner"],
    "work": ["the quarterly planning meeting", "the new project schedule", "the office move",
             "the training workshop", "my commute this week", "the feedback from my manager"],
    "family": ["my sister's birthday party", "the weekend with my grandparents", "our family board game night",
               "my cousin's wedding", "the trip to see my parents", "my son's school play"],
    "shopping": ["the sale at the shoe store", "the delivery of our new sofa", "the queue at the supermarket",
                 "the online order for the kitchen", "the bookshop downtown", "the return of the broken lamp"],
    "nature": ["the hike to the waterfall", "the birds in the garden", "the flowers along the river path",
               "the pine forest behind the house", "the sunset over the lake", "the autumn leaves in the park"],
}
TOPIC_LABELS = list(TOPICS)

SENTIMENT = {
    1: ["was a complete disaster", "was awful from start to finish", "left me furious"],
    2: ["was disappointing", "was worse than I expected", "was a bit of a letdown"],
    3: ["was fine, nothing special", "was about what I expected", "had good and bad parts"],
    4: ["was pleasant", "turned out better than expected", "was quite enjoyable"],
    5: ["was absolutely wonderful", "was the best in years", "made my whole week"],
}
FORMS = [
    "{clause} {sentiment}.",
    "Honestly, {clause} {sentiment}.",
    "I have to say {clause} {sentiment}.",
    "Did you hear that {clause} {sentiment}?",
    "Can you believe {clause} {sentiment}?",
    "Please tell me if you think {clause} {sentiment} too.",
    "Between you and me, {clause} {sentiment}, although opinions differ.",
]
HEDGES = ["I suppose", "Maybe", "To be fair", "On the other hand", "Then again", "Still"]

TASKS = {
    "binary-food": {
        "schema_version": 1,
        "name": "Food mention",
        "instructions": "Decide whether the text mentions food or drink in any way.",
        "categories": [
            {"id": "yes", "label": "mentions food or drink", "response": "A", "value": None},
            {"id": "no", "label": "does not mention food or drink", "response": "B", "value": None},
        ],
        "ordered": False,
        "examples": [],
    },
    "sentiment-5": {
        "schema_version": 1,
        "name": "Overall sentiment",
        "instructions": "Rate the overall sentiment the writer expresses.",
        "categories": [
            {"id": f"s{i}", "label": label, "response": str(i), "value": float(i)}
            for i, label in enumerate(["very negative", "negative", "neutral", "positive", "very positive"], start=1)
        ],
        "ordered": True,
        "examples": [
            {"text": "The concert was a complete disaster.", "category_id": "s1"},
            {"text": "Lunch was fine, nothing special.", "category_id": "s3"},
            {"text": "The holiday was absolutely wonderful.", "category_id": "s5"},
        ],
    },
    "topic-10": {
        "schema_version": 1,
        "name": "Main topic",
        "instructions": "Choose the single main topic of the text.",
        "categories": [
            {"id": t, "label": t, "response": chr(ord("A") + i), "value": None} for i, t in enumerate(TOPIC_LABELS)
        ],
        "ordered": False,
        "examples": [],
    },
}


def sentence(rng: random.Random, topic: str | None = None, score: int | None = None) -> str:
    topic = topic or rng.choice(TOPIC_LABELS)
    score = score or rng.randint(1, 5)
    clause = rng.choice(TOPICS[topic])
    text = rng.choice(FORMS).format(clause=clause, sentiment=rng.choice(SENTIMENT[score]))
    return text[0].upper() + text[1:]


def passage(rng: random.Random, n_sentences: int) -> str:
    main = rng.choice(TOPIC_LABELS)
    parts = []
    for i in range(n_sentences):
        topic = main if rng.random() < 0.7 else rng.choice(TOPIC_LABELS)
        s = sentence(rng, topic)
        if i and rng.random() < 0.25:
            s = f"{rng.choice(HEDGES)}, {s[0].lower()}{s[1:]}"
        parts.append(s)
    return " ".join(parts)


def build(seed: int, n_short: int, n_medium: int, n_long: int) -> list[dict]:
    rng = random.Random(seed)
    texts = []
    for i in range(n_short):
        texts.append({"id": f"short-{i:04d}", "length": "short", "text": sentence(rng)})
    for i in range(n_medium):
        texts.append({"id": f"medium-{i:04d}", "length": "medium", "text": passage(rng, rng.randint(4, 6))})
    for i in range(n_long):
        texts.append({"id": f"long-{i:04d}", "length": "long", "text": passage(rng, rng.randint(30, 45))})
    return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--short", type=int, default=300)
    ap.add_argument("--medium", type=int, default=150)
    ap.add_argument("--long", type=int, default=50)
    args = ap.parse_args()
    corpus = {
        "corpus_version": CORPUS_VERSION,
        "seed": args.seed,
        "tasks": TASKS,
        "texts": build(args.seed, args.short, args.medium, args.long),
    }
    args.out.write_text(json.dumps(corpus, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {len(corpus['texts'])} texts to {args.out}")


if __name__ == "__main__":
    main()
