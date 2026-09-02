"""
Tokyo Luxury Hotel Instagram — Hashtag Strategy Review
======================================================

A comparative diagnostic across six Tokyo luxury hotels.

Run:
    streamlit run app.py

Expects the Apify export in the same folder:
    dataset_instagram-scraper-task-3_2026-09-02_10-51-38-883.csv
"""

import re
import unicodedata
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

DATA_FILE = "dataset_instagram-scraper-task-3_2026-09-02_10-51-38-883.csv"
WINDOW_START = "2023-05-01"

HOTELS = {
    "aman_tokyo": "Aman Tokyo",
    "thepeninsulatokyo": "The Peninsula Tokyo",
    "parkhyatttokyo": "Park Hyatt Tokyo",
    "janutokyo": "Janu Tokyo",
    "thecapitolhoteltokyu": "The Capitol Hotel Tokyu",
    "ritzcarltontokyo": "The Ritz-Carlton, Tokyo",
}

# Ordered so charts read consistently everywhere.
HOTEL_ORDER = [
    "The Peninsula Tokyo",
    "Janu Tokyo",
    "Aman Tokyo",
    "Park Hyatt Tokyo",
    "The Ritz-Carlton, Tokyo",
    "The Capitol Hotel Tokyu",
]

INK = "#1B1B18"
MUTED = "#75736B"
RULE = "#DFDBD1"
PAPER = "#FBFAF7"
ACCENT = "#2E5248"        # focal hotel
NEUTRAL = "#C8C4B8"       # peer hotels

st.set_page_config(
    page_title="Tokyo Luxury Hotel Instagram Review",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,300;6..72,400;6..72,500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

      .stApp {{ background: {PAPER}; }}
      .block-container {{ max-width: 1080px; padding-top: 3rem; padding-bottom: 5rem; }}

      html, body, [class*="css"] {{
          font-family: 'IBM Plex Sans', system-ui, sans-serif;
          color: {INK};
      }}
      h1, h2, h3 {{ font-family: 'Newsreader', Georgia, serif; font-weight: 400; }}
      h1 {{ font-size: 2.45rem; line-height: 1.15; letter-spacing: -0.015em; margin-bottom: .2rem; }}
      h2 {{ font-size: 1.6rem; margin-top: 3.2rem; margin-bottom: .4rem; }}
      h3 {{ font-size: 1.15rem; margin-top: 1.6rem; }}

      p, li {{ font-size: 0.95rem; line-height: 1.62; max-width: 68ch; color: #33322D; }}

      .lede {{ font-family: 'Newsreader', Georgia, serif; font-size: 1.22rem;
              line-height: 1.55; color: {MUTED}; max-width: 62ch; margin-bottom: 1.4rem; }}
      .note {{ font-size: 0.83rem; color: {MUTED}; line-height: 1.55; max-width: 70ch; }}
      .rule {{ border-top: 1px solid {RULE}; margin: 2.6rem 0 0 0; }}

      .figure {{ font-family: 'Newsreader', Georgia, serif; font-size: 2.9rem;
                line-height: 1; color: {ACCENT}; }}
      .figure-label {{ font-size: 0.86rem; color: {MUTED}; margin-top: .35rem; max-width: 30ch; }}

      .finding {{ border-left: 2px solid {ACCENT}; padding: .1rem 0 .1rem 1.1rem;
                 margin: 1.4rem 0; max-width: 66ch; }}
      .finding strong {{ font-weight: 600; }}

      div[data-testid="stMetricValue"] {{ font-family: 'Newsreader', serif; font-weight: 400; }}
      #MainMenu, footer, header {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------------

def strip_invisible(text: str) -> str:
    """Remove Unicode format characters (Cf) such as U+2060 word joiner.

    Instagram treats a tag containing an invisible character as distinct from
    the same tag without it, so these need to be tracked, not silently dropped.
    """
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


def has_invisible(text: str) -> bool:
    return any(unicodedata.category(ch) == "Cf" for ch in text)


def is_japanese(text: str) -> bool:
    return bool(re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", text))


def find_data_file(preferred: str):
    """Locate the export next to this script.

    Streamlit Cloud does not guarantee that the working directory is the repo
    root, so a bare relative filename can miss. Look beside the script first,
    then fall back to any Apify export in the same folder.
    """
    here = Path(__file__).resolve().parent
    for candidate in (here / preferred, Path.cwd() / preferred):
        if candidate.exists():
            return candidate

    for pattern in ("dataset_instagram-scraper*.csv", "*.csv"):
        matches = sorted(here.glob(pattern))
        if matches:
            # Largest file is the full export rather than a partial run.
            return max(matches, key=lambda p: p.stat().st_size)
    return None


@st.cache_data(show_spinner="Loading dataset…")
def load_posts(path: str):
    """Return one row per post, with hashtag counts attached."""
    source = find_data_file(path)
    if source is None:
        return None, None

    raw = pd.read_csv(source, encoding="utf-8-sig", low_memory=False)

    tag_cols = [c for c in raw.columns if re.fullmatch(r"hashtags/\d+", c)]
    keep = ["ownerUsername", "timestamp", "caption", "likesCount",
            "commentsCount", "type"]
    keep = [c for c in keep if c in raw.columns]

    df = raw[keep + tag_cols].copy()
    df["account"] = df["ownerUsername"].astype(str).str.lower().str.strip()
    df = df[df["account"].isin(HOTELS)].copy()
    df["hotel"] = df["account"].map(HOTELS)

    df["date"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df[df["date"] >= WINDOW_START].copy()

    df["tag_count"] = df[tag_cols].notna().sum(axis=1)
    df["likes"] = pd.to_numeric(df["likesCount"], errors="coerce")
    df["comments"] = pd.to_numeric(df["commentsCount"], errors="coerce")
    df["format"] = df["type"].replace({"Sidecar": "Carousel"})
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    posts = df[["hotel", "date", "month", "tag_count", "likes",
                "comments", "format"]].reset_index(drop=True)

    # Long form: one row per hashtag occurrence.
    long = df.melt(
        id_vars=["hotel", "date", "likes", "comments"],
        value_vars=tag_cols,
        value_name="tag_raw",
    ).dropna(subset=["tag_raw"])

    long["tag_raw"] = long["tag_raw"].astype(str).str.strip().str.lstrip("#")
    long = long[long["tag_raw"] != ""].copy()
    long["invisible"] = long["tag_raw"].apply(has_invisible)
    long["tag"] = long["tag_raw"].apply(strip_invisible).str.lower().str.strip()
    long["script"] = np.where(long["tag"].apply(is_japanese), "Japanese", "Latin")
    long = long[long["tag"] != ""].copy()

    return posts, long.reset_index(drop=True)


def concentration(tags: pd.Series) -> dict:
    """Herfindahl index and effective vocabulary size for one hotel's tags."""
    counts = tags.value_counts()
    if counts.empty:
        return {"unique": 0, "hhi": np.nan, "top10_share": np.nan}
    share = counts / counts.sum()
    return {
        "unique": int(counts.size),
        "hhi": float((share ** 2).sum()),
        "top10_share": float(counts.head(10).sum() / counts.sum()),
    }


def spearman(x, y):
    """Rank correlation without a scipy dependency."""
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 10 or pair["x"].nunique() < 3:
        return np.nan, len(pair)
    return pair["x"].rank().corr(pair["y"].rank()), len(pair)


def bar(data, value, label, focal, value_title, fmt=".2f", height=230):
    """Horizontal bar chart with the focal hotel picked out."""
    data = data.copy()
    data["focal"] = data[label] == focal
    return (
        alt.Chart(data)
        .mark_bar(height=17)
        .encode(
            x=alt.X(f"{value}:Q", title=value_title,
                    axis=alt.Axis(grid=True, gridColor=RULE, gridDash=[2, 3],
                                  domain=False, tickSize=0, labelColor=MUTED,
                                  titleColor=MUTED, titleFontWeight="normal")),
            y=alt.Y(f"{label}:N", title=None, sort=HOTEL_ORDER,
                    axis=alt.Axis(domain=False, tickSize=0, labelColor=INK,
                                  labelFontSize=12, labelPadding=8)),
            color=alt.condition(alt.datum.focal, alt.value(ACCENT), alt.value(NEUTRAL)),
            tooltip=[alt.Tooltip(f"{label}:N", title="Hotel"),
                     alt.Tooltip(f"{value}:Q", title=value_title, format=fmt)],
        )
        .properties(height=height)
        .configure_view(strokeWidth=0)
        .configure_axis(labelFont="IBM Plex Sans", titleFont="IBM Plex Sans")
    )


# ----------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------

posts, long = load_posts(DATA_FILE)

if posts is None:
    here = Path(__file__).resolve().parent
    present = sorted(p.name for p in here.iterdir() if p.is_file())
    st.error(f"No CSV found in `{here}`.")
    st.write("Files this script can see:")
    st.code("\n".join(present) or "(the folder is empty)")
    st.stop()

window_label = (
    f"{posts['date'].min():%B %Y} – {posts['date'].max():%B %Y}"
)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------

st.markdown("# How Tokyo's luxury hotels use hashtags")
st.markdown(
    f"<p class='lede'>A comparison of {len(posts):,} Instagram posts from six "
    f"Tokyo luxury properties, {window_label}. The question is not who posts "
    "most, but what each property chooses to make findable.</p>",
    unsafe_allow_html=True,
)

focal = st.selectbox(
    "Read this from the perspective of",
    HOTEL_ORDER,
    index=HOTEL_ORDER.index("Park Hyatt Tokyo"),
)

focal_posts = posts[posts["hotel"] == focal]
focal_tags = long[long["hotel"] == focal]

st.markdown("<div class='rule'></div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 1. Volume
# ----------------------------------------------------------------------------

st.markdown("## There is no house style in this market")

volume = (
    posts.groupby("hotel")
    .agg(posts=("tag_count", "size"),
         tags=("tag_count", "sum"),
         per_post=("tag_count", "mean"),
         untagged=("tag_count", lambda s: (s == 0).mean() * 100))
    .reset_index()
)

low = volume.loc[volume["per_post"].idxmin()]
high = volume.loc[volume["per_post"].idxmax()]
focal_row = volume[volume["hotel"] == focal].iloc[0]

st.markdown(
    f"""
Tagging practice across these six properties spans a factor of
{high['per_post'] / max(low['per_post'], 0.01):.0f}. {low['hotel']} averages
{low['per_post']:.2f} hashtags per post; {high['hotel']} averages
{high['per_post']:.1f}. Both are established luxury properties in the same city
competing for overlapping guests. Whatever the right number is, the market has
not converged on it — which means the volume you have settled on is a choice,
not an industry standard.
"""
)

st.altair_chart(bar(volume, "per_post", "hotel", focal, "Hashtags per post"),
                use_container_width=True)

a, b, c = st.columns(3)
a.markdown(
    f"<div class='figure'>{focal_row['per_post']:.1f}</div>"
    f"<div class='figure-label'>hashtags per post at {focal}</div>",
    unsafe_allow_html=True)
b.markdown(
    f"<div class='figure'>{focal_row['untagged']:.0f}%</div>"
    f"<div class='figure-label'>of its posts carry no hashtag at all</div>",
    unsafe_allow_html=True)
c.markdown(
    f"<div class='figure'>{int(focal_row['posts']):,}</div>"
    f"<div class='figure-label'>posts in the sample window</div>",
    unsafe_allow_html=True)

pen = volume[volume["hotel"] == "The Peninsula Tokyo"].iloc[0]
st.markdown(
    f"""
<div class='finding'>
<strong>The Peninsula runs almost without hashtags.</strong> {pen['untagged']:.0f}%
of its {int(pen['posts']):,} posts carry none, and the account still draws a
median of {posts[posts.hotel == 'The Peninsula Tokyo']['likes'].median():.0f}
likes per post. Its tags sit in the first comment instead, which keeps the
caption clean while preserving searchability. If you have assumed hashtags are
obligatory on a luxury account, this is the counter-example in your own
competitive set.
</div>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# 2. Vocabulary discipline
# ----------------------------------------------------------------------------

st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
st.markdown("## Volume is the wrong metric. Repetition is the right one.")

st.markdown(
    """
Counting hashtags tells you how loud an account is. It says nothing about
whether the same language comes back post after post — and repetition is what
builds a searchable, recognisable brand vocabulary. Two properties can post an
identical number of tags while one rehearses a tight set and the other improvises
every time.

The measure below is a concentration index. A high value means a small number of
tags account for most usage. A low value means the vocabulary is diffuse.
"""
)

conc = pd.DataFrame([
    {"hotel": h, **concentration(long[long["hotel"] == h]["tag"])}
    for h in HOTEL_ORDER
])
conc["entries"] = [len(long[long["hotel"] == h]) for h in conc["hotel"]]

st.altair_chart(bar(conc, "hhi", "hotel", focal, "Concentration (Herfindahl)", ".3f"),
                use_container_width=True)

table = conc[["hotel", "entries", "unique", "hhi", "top10_share"]].copy()
table.columns = ["Hotel", "Hashtag uses", "Distinct tags",
                 "Concentration", "Top 10 share"]
table["Concentration"] = table["Concentration"].map("{:.3f}".format)
table["Top 10 share"] = (table["Top 10 share"] * 100).map("{:.0f}%".format)
st.dataframe(table, hide_index=True, use_container_width=True)

janu = conc[conc["hotel"] == "Janu Tokyo"].iloc[0]
cap = conc[conc["hotel"] == "The Capitol Hotel Tokyu"].iloc[0]

st.markdown(
    f"""
<div class='finding'>
<strong>Janu says the same {int(janu['unique'])} things every time.</strong>
Across {int(janu['entries'])} hashtag uses it draws on {int(janu['unique'])}
distinct tags, and its top ten account for {janu['top10_share'] * 100:.0f}% of
all usage. The Capitol Hotel Tokyu spreads {int(cap['entries']):,} uses across
{int(cap['unique']):,} distinct tags. Janu opened in 2024; that discipline is a
launch decision, and it is the cheapest one on this page to copy.
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(f"### What {focal} actually repeats")
top_tags = (
    focal_tags["tag"].value_counts().head(12)
    .rename_axis("Hashtag").reset_index(name="Uses")
)
top_tags["Share of all uses"] = (
    top_tags["Uses"] / len(focal_tags) * 100
).map("{:.1f}%".format)
st.dataframe(top_tags, hide_index=True, use_container_width=True)

# ----------------------------------------------------------------------------
# 3. Does tagging more work?
# ----------------------------------------------------------------------------

st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
st.markdown("## More hashtags does not mean more engagement")

rows = []
for hotel in HOTEL_ORDER:
    sub = posts[posts["hotel"] == hotel]
    r_likes, n = spearman(sub["tag_count"], sub["likes"])
    r_comments, _ = spearman(sub["tag_count"], sub["comments"])
    rows.append({"hotel": hotel, "likes": r_likes,
                 "comments": r_comments, "n": n})
corr = pd.DataFrame(rows)

st.markdown(
    """
Each bar is the relationship between how many hashtags a post carries and how
much engagement it receives, measured within that hotel's own account. Comparing
across hotels would only measure who has more followers, so every correlation
below is internal to one property.
"""
)

melted = corr.melt(id_vars="hotel", value_vars=["likes", "comments"],
                   var_name="metric", value_name="rho")
melted["metric"] = melted["metric"].str.title()

chart = (
    alt.Chart(melted)
    .mark_bar(height=13)
    .encode(
        x=alt.X("rho:Q", title="Rank correlation with hashtag count",
                scale=alt.Scale(domain=[-0.35, 0.35]),
                axis=alt.Axis(grid=True, gridColor=RULE, gridDash=[2, 3],
                              domain=False, tickSize=0, labelColor=MUTED,
                              titleColor=MUTED, titleFontWeight="normal")),
        y=alt.Y("hotel:N", title=None, sort=HOTEL_ORDER,
                axis=alt.Axis(domain=False, tickSize=0, labelColor=INK,
                              labelFontSize=12, labelPadding=8)),
        yOffset=alt.YOffset("metric:N"),
        color=alt.Color("metric:N",
                        scale=alt.Scale(domain=["Likes", "Comments"],
                                        range=[ACCENT, NEUTRAL]),
                        legend=alt.Legend(title=None, orient="top",
                                          labelColor=MUTED)),
        tooltip=["hotel", "metric", alt.Tooltip("rho:Q", format=".3f")],
    )
    .properties(height=260)
    .configure_view(strokeWidth=0)
    .configure_axis(labelFont="IBM Plex Sans", titleFont="IBM Plex Sans")
)
st.altair_chart(chart, use_container_width=True)

aman_c = corr[corr["hotel"] == "Aman Tokyo"]["comments"].iloc[0]
st.markdown(
    f"""
<div class='finding'>
<strong>The pattern splits by account, and it splits by metric.</strong>
At Janu, Park Hyatt and the Capitol, posts with more hashtags attract modestly
more likes. At Aman and the Ritz-Carlton the effect on likes is flat — and at
Aman, heavily tagged posts draw measurably <em>fewer</em> comments
(rho = {aman_c:.2f}). Reach and conversation are not the same outcome, and
adding tags appears to trade one for the other on the most established accounts.
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "<p class='note'>These are correlations across observational data, not "
    "experiments. A post's subject, timing and format all move engagement too, "
    "and none of them are controlled here.</p>",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# 4. Format
# ----------------------------------------------------------------------------

st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
st.markdown("## Video is the weakest format in this set")

fmt = (
    posts.groupby("format")
    .agg(posts=("likes", "size"), median_likes=("likes", "median"),
         median_comments=("comments", "median"))
    .reset_index().sort_values("median_likes", ascending=False)
)

fmt_chart = (
    alt.Chart(fmt)
    .mark_bar(height=26)
    .encode(
        x=alt.X("median_likes:Q", title="Median likes",
                axis=alt.Axis(grid=True, gridColor=RULE, gridDash=[2, 3],
                              domain=False, tickSize=0, labelColor=MUTED,
                              titleColor=MUTED, titleFontWeight="normal")),
        y=alt.Y("format:N", title=None, sort="-x",
                axis=alt.Axis(domain=False, tickSize=0, labelColor=INK,
                              labelFontSize=12, labelPadding=8)),
        color=alt.value(ACCENT),
        tooltip=["format", "posts", "median_likes", "median_comments"],
    )
    .properties(height=150)
    .configure_view(strokeWidth=0)
    .configure_axis(labelFont="IBM Plex Sans", titleFont="IBM Plex Sans")
)
st.altair_chart(fmt_chart, use_container_width=True)

mix = pd.crosstab(posts["hotel"], posts["format"], normalize="index") * 100
focal_video = mix.loc[focal, "Video"] if "Video" in mix.columns else 0

st.markdown(
    f"""
Single images draw a median of {fmt[fmt.format == 'Image']['median_likes'].iloc[0]:.0f}
likes, carousels {fmt[fmt.format == 'Carousel']['median_likes'].iloc[0]:.0f}, and
video {fmt[fmt.format == 'Video']['median_likes'].iloc[0]:.0f}. That inverts the
usual advice, and it inverts published findings from hotel Instagram research in
other markets. {focal} currently runs {focal_video:.0f}% video.

One caveat worth stating plainly: this sample covers feed posts only. Reels are
excluded, so this compares video *within the feed* against stills, not the whole
video strategy.
"""
)

st.dataframe(
    mix.round(1).reset_index().rename(columns={"hotel": "Hotel"}),
    hide_index=True, use_container_width=True,
)

# ----------------------------------------------------------------------------
# 5. Language
# ----------------------------------------------------------------------------

st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
st.markdown("## Who each property is tagging for")

script = pd.crosstab(long["hotel"], long["script"], normalize="index") * 100
script = script.reset_index()

script_chart = (
    alt.Chart(script.melt(id_vars="hotel", var_name="script", value_name="pct"))
    .mark_bar(height=20)
    .encode(
        x=alt.X("pct:Q", stack="normalize", title=None,
                axis=alt.Axis(format="%", grid=False, domain=False,
                              tickSize=0, labelColor=MUTED)),
        y=alt.Y("hotel:N", title=None, sort=HOTEL_ORDER,
                axis=alt.Axis(domain=False, tickSize=0, labelColor=INK,
                              labelFontSize=12, labelPadding=8)),
        color=alt.Color("script:N",
                        scale=alt.Scale(domain=["Japanese", "Latin"],
                                        range=[ACCENT, NEUTRAL]),
                        legend=alt.Legend(title=None, orient="top",
                                          labelColor=MUTED)),
        tooltip=["hotel", "script", alt.Tooltip("pct:Q", format=".1f")],
    )
    .properties(height=230)
    .configure_view(strokeWidth=0)
    .configure_axis(labelFont="IBM Plex Sans", titleFont="IBM Plex Sans")
)
st.altair_chart(script_chart, use_container_width=True)

jp_share = script.set_index("hotel")["Japanese"]
st.markdown(
    f"""
Most of the set sits between {jp_share.drop('Janu Tokyo').min():.0f}% and
{jp_share.max():.0f}% Japanese-script hashtags — a roughly even split between the
domestic guest and the inbound traveller. Janu is the outlier at
{jp_share['Janu Tokyo']:.0f}%, tagging almost entirely in Latin script.

This matters more in Tokyo than it would elsewhere. Japanese Instagram users
search by hashtag at a far higher rate than the global average, so the script you
tag in determines which of the two audiences can find the post at all. For most
properties here it is an even hedge. It is worth knowing whether that was decided
or inherited.
"""
)

# ----------------------------------------------------------------------------
# 6. Data hygiene
# ----------------------------------------------------------------------------

invisible = long[long["invisible"]]
if len(invisible) > 0:
    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
    st.markdown("## One account is splitting its own brand tag in half")

    by_hotel = invisible.groupby("hotel").size().sort_values(ascending=False)
    worst = by_hotel.index[0]
    worst_n = int(by_hotel.iloc[0])
    worst_total = len(long[long["hotel"] == worst])

    st.markdown(
        f"""
{worst_n} of {worst}'s {worst_total:,} hashtag uses
({worst_n / worst_total * 100:.0f}%) contain an invisible Unicode character —
U+2060, a word joiner — appended to the end of the tag. It is not visible in the
caption, and it almost certainly arrives by pasting copy from a document or
design file rather than typing it.

The consequence is that the same brand tag is being published as two different
tags. Below, each pair reads identically on screen but is stored, and searched,
separately.
"""
    )

    w = long[(long["hotel"] == worst)].copy()
    pairs = []
    for tag in w[w["invisible"]]["tag"].value_counts().head(6).index:
        clean_n = int(((w["tag"] == tag) & (~w["invisible"])).sum())
        dirty_n = int(((w["tag"] == tag) & (w["invisible"])).sum())
        if clean_n and dirty_n:
            pairs.append({
                "Hashtag": f"#{tag}",
                "Typed normally": clean_n,
                "With hidden character": dirty_n,
                "Share going to the broken version":
                    f"{dirty_n / (clean_n + dirty_n) * 100:.0f}%",
            })
    if pairs:
        st.dataframe(pd.DataFrame(pairs), hide_index=True,
                     use_container_width=True)

    st.markdown(
        f"""
<div class='finding'>
<strong>What to do about it.</strong> Search each of those tags on Instagram, once
typed by hand and once pasted from your caption template. If the two return
different result sets, the tag is being fragmented and roughly half of
{worst}'s brand-tag usage is landing somewhere no one searches. The fix is to
retype the hashtag block once, save it as plain text, and stop pasting it from
formatted sources.
</div>
""",
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# Method
# ----------------------------------------------------------------------------

st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
st.markdown("## How this was built")

st.markdown(
    f"""
<p class='note'>
{len(posts):,} feed posts from six Tokyo luxury hotel Instagram accounts,
published {window_label}, collected from public profile pages via the Apify
Instagram Scraper. Hashtags are read from post captions; reels and stories are
not included, and neither are hashtags placed in comments, which means figures
for accounts that tag in the first comment understate their true usage.
</p>
<p class='note'>
Engagement is likes and comments as displayed publicly. Follower counts are not
available through this method, so no cross-account engagement rate is computed
and all correlations are calculated within a single account. Every relationship
shown is observational and correlational; none of it establishes cause.
</p>
<p class='note'>
The Four Seasons Hotel Tokyo at Marunouchi was part of the original design and
was excluded: its account's public history begins in February 2026, leaving no
overlap with the study window.
</p>
""",
    unsafe_allow_html=True,
)
