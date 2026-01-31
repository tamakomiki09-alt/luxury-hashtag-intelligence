import pandas as pd
import re

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("luxury_hotel_data_tokyo.csv", encoding="latin1")
df.columns = df.columns.str.strip()

# Standardize hotel column
if "ownerUsername" in df.columns:
    df = df.rename(columns={"ownerUsername": "hotel"})

# Timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"])

# Engagement
df["engagement"] = df["likesCount"].fillna(0) + df["commentsCount"].fillna(0)

# -----------------------------
# COMBINE HASHTAG COLUMNS
# -----------------------------
hashtag_cols = [c for c in df.columns if c.startswith("hashtag")]

def combine_hashtags(row):
    tags = []
    for col in hashtag_cols:
        val = row[col]
        if pd.notna(val):
            tags.append(str(val).lower())
    return " ".join(tags)

df["hashtags_text"] = df.apply(combine_hashtags, axis=1)

# Caption cleanup
df["caption"] = df["caption"].fillna("").str.lower()

# -----------------------------
# RULE-BASED THEMES
# -----------------------------
THEME_RULES = {
    "Brand": [
        "aman", "ritz", "four seasons", "peninsula",
        "luxury", "signature", "hospitality"
    ],
    "Experience": [
        "spa", "suite", "room", "pool", "dining",
        "restaurant", "bar", "stay", "experience"
    ],
    "Seasonal": [
        "spring", "summer", "autumn", "fall", "winter",
        "sakura", "christmas", "newyear", "holiday"
    ]
}

def classify_theme(row):
    text = row["caption"] + " " + row["hashtags_text"]
    for theme, keywords in THEME_RULES.items():
        for kw in keywords:
            if kw in text:
                return theme
    return "Other"

df["theme_rule"] = df.apply(classify_theme, axis=1)

# -----------------------------
# SAVE CLEAN FILE
# -----------------------------
df.to_csv("hotel_posts_themes_ready.csv", index=False)

print("✅ Saved: hotel_posts_themes_ready.csv")
print(df["theme_rule"].value_counts())
