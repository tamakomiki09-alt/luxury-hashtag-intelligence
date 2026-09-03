"""
Hashtag classification
======================

Classifies every distinct hashtag in the dataset against a fixed codebook,
using an LLM as the coder, and writes the result to a CSV you commit to the
repo. The dashboard reads that file. It never calls the API itself.

Running classification once and committing the output matters for two reasons:
the paper needs a fixed, inspectable classification rather than one that
changes each time the app loads, and a reviewer needs to be able to see every
label you assigned.

Usage
-----
    export OPENAI_API_KEY="sk-..."
    python classify_hashtags.py                 # classify everything
    python classify_hashtags.py --sample 300    # write a blank sheet to hand-code
    python classify_hashtags.py --agreement     # score your coding vs the model

Outputs
-------
    hashtag_categories.csv      tag, category, confidence, uses
    validation_sample.csv       blank column for you to fill in by hand
    (agreement is printed to the terminal)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "hashtag_categories.csv"
VALIDATION = HERE / "validation_sample.csv"

HOTELS = {
    "aman_tokyo": "Aman Tokyo",
    "thepeninsulatokyo": "The Peninsula Tokyo",
    "parkhyatttokyo": "Park Hyatt Tokyo",
    "janutokyo": "Janu Tokyo",
    "thecapitolhoteltokyu": "The Capitol Hotel Tokyu",
    "ritzcarltontokyo": "The Ritz-Carlton, Tokyo",
}
WINDOW_START = "2023-05-01"

# ---------------------------------------------------------------------------
# Codebook
#
# Six substantive categories plus one residual. The residual is a real option,
# not a fallback for anything the coder failed to match — that distinction is
# what makes "unclassifiable" interpretable in the results.
# ---------------------------------------------------------------------------

CODEBOOK = {
    "Property identity": (
        "The hotel's own name or account handle, in any script or "
        "abbreviation. The bare identifier, not a slogan. Examples: amantokyo, "
        "ザキャピトルホテル東急, ritzcarltontokyo, parkhyatt, rctokyo, "
        "capitolhoteltokyo, リッツカールトン東京."
    ),
    "Group and loyalty": (
        "The parent company, sister properties, a collection the hotel belongs "
        "to, or a loyalty programme. Points outward to an affiliation rather "
        "than to this property. Examples: amanresorts, worldofhyatt, "
        "thepreferredlife, ipreferrewards, amanessentials, hyatt."
    ),
    "Brand philosophy": (
        "A coined, proprietary phrase expressing what the brand stands for or "
        "promises — a campaign line or positioning statement, not a name and "
        "not a description of a thing. Examples: luxuryispersonal, rcmemories, "
        "welcometojanu, 魂を呼び覚ます, thespiritofaman, capitolmoments, "
        "believeintravel, rekindlethesoul."
    ),
    "Named outlet and talent": (
        "A specific restaurant, bar, lounge, patisserie, spa or shop inside the "
        "hotel, or a named chef, designer or artist associated with it. A "
        "proper noun you could book or ask for. Examples: theloungebyaman, "
        "arvabyaman, ザカフェbyアマン, keikobayashi, 小林圭, "
        "heritagebykeikobayashi, amanspa, ラパティスリーbyアマン東京."
    ),
    "Occasion and service": (
        "A generic service, facility or dining occasion, named as a type rather "
        "than as a specific outlet. This is the category for meal-occasion "
        "tagging. Examples: afternoontea, アフタヌーンティー, ヌン活, ホテルディナー, "
        "ホテルランチ, clublounge, ウェルネス, スパ, 朝食, ホテルステイ, パフェ."
    ),
    "Japanese cultural register": (
        "Invokes Japanese cultural practice, craft, cuisine tradition, or a "
        "cultural site — signalling authenticity or rootedness in Japan rather "
        "than naming a location or a sellable service. Examples: omotenashi, "
        "おもてなし, ikebana, 生け花, 草月, 日枝神社, 日本料理, japanesecuisine, "
        "ロビー装花."
    ),
    "Season and limited time": (
        "Tied to a season, holiday, festival, anniversary or explicitly "
        "time-bound offer. Examples: クリスマス, christmas, 桜, sakura, 春, 秋, "
        "期間限定, クリスマスケーキ, pht30周年, ジャヌ東京1周年, バレンタインデー."
    ),
    "Location": (
        "A geographic place: country, city, ward, neighbourhood, district, "
        "landmark or transit hub. Examples: 赤坂, 溜池山王, tokyo, japan, "
        "marunouchi, 東京ミッドタウン, 大手町の森, ginza."
    ),
    "Discovery stack": (
        "A broad, generic search term aimed at reach, describing a category of "
        "hotel, travel or food with no brand, outlet or specific place named. "
        "Often high-volume and used by several competitors alike. Examples: "
        "luxurytokyo, luxuryjapan, visitjapan, ラグジュアリーホテル, 東京ホテル, "
        "tokyohotel, 東京グルメ, tokyofoodie, 東京観光, luxurystay."
    ),
    "Unclassifiable": (
        "Emoji only, an unrelated personal handle, gibberish, or a fragment "
        "whose meaning cannot be determined without seeing the post. Use "
        "sparingly and only when a genuine reading is impossible."
    ),
}

CATEGORIES = list(CODEBOOK)

SYSTEM_PROMPT = (
    "You are coding Instagram hashtags used by luxury hotels in Tokyo for an "
    "academic content analysis. Tags may be in Japanese, English, or romanised "
    "Japanese. Assign each tag to exactly one category from the codebook.\n\n"
    "Codebook:\n"
    + "\n".join(f"- {name}: {definition}" for name, definition in CODEBOOK.items())
    + "\n\nRules:\n"
    "- Assign the single best-fitting category. Do not invent categories.\n"
    "- Judge the tag on its own terms; you are not shown the post.\n"
    "- A brand name inside a longer phrase still makes the tag Brand.\n"
    "- A place name used as pure discovery bait (e.g. tokyotrip) is Place, "
    "not Generic reach, when a real location is named.\n"
    "- Give a confidence of high, medium or low. Use low when the tag is "
    "ambiguous between two categories.\n"
    "- Return only JSON. No commentary, no markdown fences."
)

BATCH_SIZE = 60
MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def strip_invisible(text: str) -> str:
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


def find_data_file() -> Path:
    for pattern in ("dataset_instagram-scraper*.csv", "*.csv"):
        matches = [p for p in sorted(HERE.glob(pattern))
                   if p.name not in {OUTPUT.name, VALIDATION.name}]
        if matches:
            return max(matches, key=lambda p: p.stat().st_size)
    sys.exit(f"No CSV found in {HERE}")


def load_tags() -> pd.DataFrame:
    """Return every distinct hashtag with how often it is used."""
    raw = pd.read_csv(find_data_file(), encoding="utf-8-sig", low_memory=False)
    tag_cols = [c for c in raw.columns if re.fullmatch(r"hashtags/\d+", c)]

    raw["account"] = raw["ownerUsername"].astype(str).str.lower().str.strip()
    df = raw[raw["account"].isin(HOTELS)].copy()
    df["date"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df[df["date"] >= WINDOW_START]

    long = df.melt(id_vars=["account"], value_vars=tag_cols,
                   value_name="tag").dropna(subset=["tag"])
    long["tag"] = (long["tag"].astype(str).str.strip().str.lstrip("#")
                   .apply(strip_invisible).str.lower().str.strip())
    long = long[long["tag"] != ""]

    counts = long["tag"].value_counts().rename_axis("tag").reset_index(name="uses")
    return counts


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_batch(client, tags: list[str]) -> dict:
    payload = json.dumps(tags, ensure_ascii=False)
    user = (
        "Classify each hashtag below. Return a JSON object mapping each tag "
        'exactly as given to {"category": "...", "confidence": "..."}.\n\n'
        f"{payload}"
    )
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
    )
    return json.loads(response.choices[0].message.content)


def run_classification() -> None:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("Set OPENAI_API_KEY first:  export OPENAI_API_KEY='sk-...'")

    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("pip install openai")

    client = OpenAI(api_key=key)
    tags = load_tags()

    # Resume support: a long run should not have to start over.
    done = {}
    if OUTPUT.exists():
        prior = pd.read_csv(OUTPUT)
        done = dict(zip(prior["tag"], zip(prior["category"], prior["confidence"])))
        print(f"Resuming — {len(done):,} tags already classified.")

    todo = [t for t in tags["tag"] if t not in done]
    print(f"{len(tags):,} distinct tags, {len(todo):,} to classify.")

    for start in range(0, len(todo), BATCH_SIZE):
        batch = todo[start:start + BATCH_SIZE]
        try:
            result = classify_batch(client, batch)
        except Exception as exc:                      # noqa: BLE001
            print(f"  batch failed ({exc}); retrying once in 5s")
            time.sleep(5)
            try:
                result = classify_batch(client, batch)
            except Exception as exc2:                 # noqa: BLE001
                print(f"  batch failed again ({exc2}); skipping")
                continue

        for tag in batch:
            entry = result.get(tag) or {}
            category = entry.get("category", "Unclassifiable")
            if category not in CATEGORIES:
                category = "Unclassifiable"
            done[tag] = (category, entry.get("confidence", "low"))

        pct = min(start + BATCH_SIZE, len(todo)) / len(todo) * 100
        print(f"  {pct:5.1f}%  ({len(done):,} coded)")

        # Write after every batch so an interrupted run loses nothing.
        out = tags.copy()
        out["category"] = out["tag"].map(lambda t: done.get(t, (None, None))[0])
        out["confidence"] = out["tag"].map(lambda t: done.get(t, (None, None))[1])
        out.dropna(subset=["category"]).to_csv(OUTPUT, index=False)

    print(f"\nWrote {OUTPUT.name}")
    final = pd.read_csv(OUTPUT)
    print("\nBy category (distinct tags / total uses):")
    summary = final.groupby("category").agg(
        tags=("tag", "size"), uses=("uses", "sum")).sort_values("uses", ascending=False)
    print(summary.to_string())
    low = (final["confidence"] == "low").mean() * 100
    print(f"\nModel reported low confidence on {low:.1f}% of tags.")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def write_validation_sample(n: int) -> None:
    """Draw a stratified random sample for hand-coding.

    Stratifying by usage band matters: a simple random sample of distinct tags
    is dominated by tags used once or twice, which is not where the model's
    errors would do the most damage.
    """
    tags = load_tags()
    tags["band"] = pd.cut(tags["uses"], bins=[0, 1, 5, 25, 10 ** 9],
                          labels=["1 use", "2-5", "6-25", "26+"])

    per_band = max(1, n // tags["band"].nunique())
    parts = []
    for band, group in tags.groupby("band", observed=True):
        take = group.sample(min(per_band, len(group)), random_state=7).copy()
        take["band"] = band
        parts.append(take)
    sample = pd.concat(parts, ignore_index=True)

    sample = sample.sample(frac=1, random_state=7).reset_index(drop=True)
    sheet = sample[["tag", "uses", "band"]].copy()
    sheet["my_category"] = ""                                # you fill this in
    sheet.to_csv(VALIDATION, index=False)

    print(f"Wrote {VALIDATION.name} — {len(sheet)} tags to code by hand.\n")
    print("Fill in my_category using exactly one of:")
    for name in CATEGORIES:
        print(f"  {name}")
    print("\nCode them before you look at hashtag_categories.csv, or the "
          "agreement figure is worthless.")


def score_agreement() -> None:
    if not VALIDATION.exists():
        sys.exit("Run --sample first, then code the sheet by hand.")
    if not OUTPUT.exists():
        sys.exit("Run the classifier first.")

    mine = pd.read_csv(VALIDATION)
    mine = mine[mine["my_category"].notna() & (mine["my_category"] != "")]
    if mine.empty:
        sys.exit("validation_sample.csv has no codes filled in yet.")

    model = pd.read_csv(OUTPUT)[["tag", "category"]]
    merged = mine.merge(model, on="tag", how="inner")
    merged["match"] = (merged["my_category"].str.strip().str.lower()
                       == merged["category"].str.strip().str.lower())

    agreement = merged["match"].mean() * 100
    print(f"Hand-coded {len(merged)} tags.")
    print(f"Agreement with the model: {agreement:.1f}%\n")

    # Cohen's kappa — percent agreement alone flatters a skewed distribution.
    observed = merged["match"].mean()
    p_mine = merged["my_category"].value_counts(normalize=True)
    p_model = merged["category"].value_counts(normalize=True)
    expected = sum(p_mine.get(c, 0) * p_model.get(c, 0) for c in CATEGORIES)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else float("nan")
    print(f"Cohen's kappa: {kappa:.3f}")
    print("(above 0.80 is strong, 0.61-0.80 substantial, below 0.60 needs work)\n")

    wrong = merged[~merged["match"]]
    if not wrong.empty:
        print(f"Disagreements ({len(wrong)}):")
        print(wrong[["tag", "my_category", "category"]].to_string(index=False))
        print("\nRead these before you write up. A pattern in the "
              "disagreements usually means a codebook definition is unclear, "
              "not that the model is wrong.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, metavar="N",
                        help="write a blank sheet of N tags to hand-code")
    parser.add_argument("--agreement", action="store_true",
                        help="score your hand-coding against the model")
    args = parser.parse_args()

    if args.sample:
        write_validation_sample(args.sample)
    elif args.agreement:
        score_agreement()
    else:
        run_classification()
