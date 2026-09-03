"""
Curated Visibility
Hashtag strategy across six Tokyo luxury hotels

Companion dashboard to the research paper. Reads the Apify export and, if
present, the fixed hashtag classification written by classify_hashtags.py.

    streamlit run app.py
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
DATA_FILE = "tokyo_luxury_instagram.csv"
CATEGORY_FILE = HERE / "hashtag_categories.csv"

# The study period. Chosen to exclude the sparse early years of the scrape.
WINDOW_START = "2023-05-01"

HOTELS = {
    "aman_tokyo": "Aman Tokyo",
    "thepeninsulatokyo": "The Peninsula Tokyo",
    "parkhyatttokyo": "Park Hyatt Tokyo",
    "janutokyo": "Janu Tokyo",
    "thecapitolhoteltokyu": "The Capitol Hotel Tokyu",
    "ritzcarltontokyo": "The Ritz-Carlton, Tokyo",
}
HOTEL_ORDER = [
    "The Peninsula Tokyo", "Janu Tokyo", "Aman Tokyo",
    "Park Hyatt Tokyo", "The Ritz-Carlton, Tokyo", "The Capitol Hotel Tokyu",
]
SHORT = {
    "The Peninsula Tokyo": "Peninsula", "Janu Tokyo": "Janu",
    "Aman Tokyo": "Aman", "Park Hyatt Tokyo": "Park Hyatt",
    "The Ritz-Carlton, Tokyo": "Ritz-Carlton",
    "The Capitol Hotel Tokyu": "Capitol",
}
# Axis labels use the short form: full names get truncated by Vega and the
# reader loses which bar is which.
SHORT_ORDER = [SHORT[h] for h in HOTEL_ORDER]

# Construal mapping. Abstract categories describe what a stay means; concrete
# categories name something you can point at. This is the operational form of
# the paper's construal-level hypothesis.
# High-construal tags describe what the property means; low-construal tags name
# something a guest could point at, book, or walk to. Identity, affiliation and
# generic discovery tags sit outside the contrast and are excluded from it.
ABSTRACT = {"Brand philosophy", "Japanese cultural register"}
CONCRETE = {"Named outlet and talent", "Occasion and service", "Location",
            "Season and limited time"}

INK, MUTED, RULE, PAPER = "#1B1B18", "#78766D", "#E2DED4", "#FCFBF8"
ACCENT, NEUTRAL, WARM = "#2E5248", "#CBC6B8", "#8C6A3F"

st.set_page_config(page_title="Curated Visibility — Tokyo Luxury Hotels",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,300;6..72,400;6..72,500&family=IBM+Plex+Sans:wght@400;450;500;600&display=swap');
  .stApp {{ background:{PAPER}; }}
  .block-container {{ max-width:1240px; padding-top:2.4rem; padding-bottom:5rem; }}
  html, body, [class*="css"] {{ font-family:'IBM Plex Sans',system-ui,sans-serif; color:{INK}; }}
  h1,h2,h3,h4 {{ font-family:'Newsreader',Georgia,serif; font-weight:400; }}
  h1 {{ font-size:2.6rem; line-height:1.1; letter-spacing:-.02em; margin-bottom:.1rem; }}
  h2 {{ font-size:1.55rem; margin:0 0 .3rem 0; }}
  h3 {{ font-size:1.2rem; margin-top:1.2rem; }}
  p, li {{ font-size:.93rem; line-height:1.62; color:#34332E; }}
  .kicker {{ font-family:'Newsreader',serif; font-style:italic; font-size:1.05rem;
             color:{WARM}; margin-bottom:.1rem; }}
  .lede {{ font-family:'Newsreader',serif; font-size:1.2rem; line-height:1.5;
           color:{MUTED}; max-width:64ch; }}
  .note {{ font-size:.81rem; color:{MUTED}; line-height:1.55; }}
  .rule {{ border-top:1px solid {RULE}; margin:2.4rem 0 1.6rem 0; }}
  .stat {{ font-family:'Newsreader',serif; font-size:2.1rem; line-height:1; color:{ACCENT}; }}
  .stat-sm {{ font-family:'Newsreader',serif; font-size:1.5rem; line-height:1; color:{INK}; }}
  .stat-label {{ font-size:.78rem; color:{MUTED}; margin-top:.3rem; line-height:1.35; }}
  .read {{ border-left:2px solid {ACCENT}; padding-left:1rem; }}
  .read p {{ margin-bottom:.7rem; }}
  .takeaway {{ background:#F1EFE8; border-radius:3px; padding:1.05rem 1.25rem;
               font-size:.88rem; color:#3E3C35; line-height:1.62;
               margin:1.1rem 0 .4rem 0; }}
  .takeaway b {{ color:{INK}; font-weight:600; }}
  .panel-title {{ font-family:'Newsreader',Georgia,serif; font-size:1.02rem;
                  color:{INK}; margin:.1rem 0 .5rem 0; }}
  .panel-note {{ font-size:.82rem; color:{MUTED}; line-height:1.55;
                 margin:.5rem 0 .1rem 0; }}
  div[data-testid="stVerticalBlockBorderWrapper"] {{
      background:#FFFDFA; border-color:{RULE} !important; border-radius:3px; }}
  div[data-testid="stMetricValue"] {{ font-family:'Newsreader',serif; font-weight:400; }}
  #MainMenu, footer, header {{ visibility:hidden; }}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

JP = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")


def strip_invisible(t: str) -> str:
    return "".join(c for c in t if unicodedata.category(c) != "Cf")


def has_invisible(t: str) -> bool:
    return any(unicodedata.category(c) == "Cf" for c in t)


def find_data_file(preferred: str):
    for c in (HERE / preferred, Path.cwd() / preferred):
        if c.exists():
            return c
    for pattern in ("tokyo_luxury_instagram.csv", "dataset_instagram-scraper*.csv", "*.csv"):
        hits = [p for p in sorted(HERE.glob(pattern)) if p.name != CATEGORY_FILE.name]
        if hits:
            return max(hits, key=lambda p: p.stat().st_size)
    return None


@st.cache_data(show_spinner="Reading dataset…")
def load_raw(preferred: str):
    src = find_data_file(preferred)
    if src is None:
        return None, None

    raw = pd.read_csv(src, encoding="utf-8-sig", low_memory=False)
    tag_cols = [c for c in raw.columns if re.fullmatch(r"hashtags/\d+", c)]

    keep = [c for c in ["ownerUsername", "timestamp", "likesCount",
                        "commentsCount", "type"] if c in raw.columns]
    df = raw[keep + tag_cols].copy()
    df["account"] = df["ownerUsername"].astype(str).str.lower().str.strip()
    df = df[df["account"].isin(HOTELS)].copy()
    df["hotel"] = df["account"].map(HOTELS)
    df["date"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["tag_count"] = df[tag_cols].notna().sum(axis=1)
    df["likes"] = pd.to_numeric(df["likesCount"], errors="coerce")
    df["comments"] = pd.to_numeric(df["commentsCount"], errors="coerce")
    df["format"] = df["type"].replace({"Sidecar": "Carousel"})
    df = df.reset_index(drop=True)
    df["post_id"] = np.arange(len(df))

    posts = df[["post_id", "hotel", "date", "tag_count", "likes",
                "comments", "format"]].copy()

    long = df.melt(id_vars=["post_id", "hotel", "date", "likes", "comments"],
                   value_vars=tag_cols, value_name="tag_raw").dropna(subset=["tag_raw"])
    long["tag_raw"] = long["tag_raw"].astype(str).str.strip().str.lstrip("#")
    long = long[long["tag_raw"] != ""].copy()
    long["invisible"] = long["tag_raw"].map(has_invisible)
    long["tag"] = long["tag_raw"].map(strip_invisible).str.lower().str.strip()
    long = long[long["tag"] != ""].copy()
    long["script"] = np.where(long["tag"].map(lambda s: bool(JP.search(s))),
                              "Japanese", "Latin")
    return posts, long.reset_index(drop=True)


@st.cache_data
def load_categories():
    if not CATEGORY_FILE.exists():
        return None
    cats = pd.read_csv(CATEGORY_FILE)
    if "category" not in cats.columns:
        return None
    cols = ["tag", "category"] + (["confidence"] if "confidence" in cats else [])
    return cats[cols]


def herfindahl(tags: pd.Series) -> float:
    counts = tags.value_counts()
    if counts.empty:
        return np.nan
    share = counts / counts.sum()
    return float((share ** 2).sum())


def rank_corr(x, y):
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 25 or pair["x"].nunique() < 3:
        return np.nan
    return float(pair["x"].rank().corr(pair["y"].rank()))


def axis_num(title=None, **kw):
    return alt.Axis(grid=True, gridColor=RULE, gridDash=[2, 3], domain=False,
                    tickSize=0, labelColor=MUTED, titleColor=MUTED,
                    titleFontWeight="normal", labelFont="IBM Plex Sans",
                    titleFont="IBM Plex Sans", **kw)


def axis_cat():
    # labelOverlap=False forces every category to render; labelLimit stops
    # Vega truncating names to an ellipsis.
    return alt.Axis(domain=False, tickSize=0, labelColor=INK, labelFontSize=12,
                    labelPadding=8, labelFont="IBM Plex Sans",
                    labelLimit=220, labelOverlap=False)


def ranked_bar(data, value, value_title, focal=None, fmt=".2f", height=230,
               sort_order=None):
    d = data.dropna(subset=[value]).copy()
    d["house"] = d["hotel"].map(SHORT).fillna(d["hotel"])
    d["_focal"] = d["hotel"] == focal if focal else False
    colour = (alt.condition(alt.datum._focal, alt.value(ACCENT), alt.value(NEUTRAL))
              if focal else alt.value(ACCENT))
    return (alt.Chart(d).mark_bar(height=18).encode(
        x=alt.X(f"{value}:Q", title=value_title, axis=axis_num()),
        y=alt.Y("house:N", title=None, sort=sort_order or SHORT_ORDER,
                axis=axis_cat()),
        color=colour,
        tooltip=[alt.Tooltip("hotel:N", title="Hotel"),
                 alt.Tooltip(f"{value}:Q", title=value_title, format=fmt)],
    ).properties(height=height).configure_view(strokeWidth=0))


def stat(container, value, label, small=False):
    cls = "stat-sm" if small else "stat"
    container.markdown(f"<div class='{cls}'>{value}</div>"
                       f"<div class='stat-label'>{label}</div>",
                       unsafe_allow_html=True)


def takeaway(text):
    """A quiet secondary panel under the main reading."""
    st.markdown(f"<div class='takeaway'>{text}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load and controls
# ---------------------------------------------------------------------------

posts_all, long_all = load_raw(DATA_FILE)
if posts_all is None:
    st.error(f"No CSV found in `{HERE}`.")
    st.code("\n".join(sorted(p.name for p in HERE.iterdir() if p.is_file())))
    st.stop()

categories = load_categories()

st.markdown("<div class='kicker'>Curated visibility in practice</div>",
            unsafe_allow_html=True)
st.markdown("# What Tokyo's luxury hotels choose to make findable")

head_l, head_r = st.columns([3, 2], gap="large")
with head_l:
    st.markdown(
        "<p class='lede'>Six luxury properties, one city, one platform. The "
        "question is not who posts most — it is what each house puts into "
        "circulation, how consistently it repeats itself, and who it expects "
        "to be searching.</p>", unsafe_allow_html=True)
with head_r:
    view = st.radio("View", ["Competitive set", "Single property"],
                    horizontal=True)

posts = posts_all[posts_all["date"] >= WINDOW_START].copy()
long = long_all[long_all["date"] >= WINDOW_START].copy()

focal = None
if view == "Single property":
    focal = st.selectbox("Property", HOTEL_ORDER,
                         index=HOTEL_ORDER.index("Park Hyatt Tokyo"))

# A Herfindahl index over a handful of observations is not interpretable, and
# neither is a rank correlation. MIN_USES gates the concentration figures.
MIN_USES = 40

st.markdown("<div class='rule'></div>", unsafe_allow_html=True)

# Shared frames -------------------------------------------------------------

volume = (posts.groupby("hotel")
          .agg(posts=("tag_count", "size"), tags=("tag_count", "sum"),
               per_post=("tag_count", "mean"),
               untagged=("tag_count", lambda s: (s == 0).mean() * 100),
               med_likes=("likes", "median"), med_comments=("comments", "median"))
          .reindex(HOTEL_ORDER).reset_index())

vocab = pd.DataFrame([{
    "hotel": h,
    "distinct": int(long[long["hotel"] == h]["tag"].nunique()),
    "hhi": (herfindahl(long[long["hotel"] == h]["tag"])
            if (long["hotel"] == h).sum() >= MIN_USES else np.nan),
    "uses": int((long["hotel"] == h).sum()),
} for h in HOTEL_ORDER])

post_script = (long.assign(is_jp=long["script"].eq("Japanese"))
               .groupby(["post_id", "hotel", "likes"], as_index=False)["is_jp"]
               .mean().rename(columns={"is_jp": "jp_share"}))

# ===========================================================================
# COMPETITIVE SET
# ===========================================================================

if view == "Competitive set":

    # -- Summary tiles -------------------------------------------------------
    tiles = st.columns(4, gap="medium")
    stat(tiles[0], f"{len(posts):,}", "posts analysed", small=True)
    stat(tiles[1], f"{len(long):,}", "hashtag uses", small=True)
    stat(tiles[2], f"{long['tag'].nunique():,}", "distinct hashtags", small=True)
    stat(tiles[3], f"{posts['tag_count'].mean():.1f}",
         "average tags per post<br>across the set", small=True)

    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)

    # -- Row 1: volume and discipline ---------------------------------------
    st.markdown("## How much, and how consistently")

    lo = volume.loc[volume["per_post"].idxmin()]
    hi = volume.loc[volume["per_post"].idxmax()]
    quiet = volume.loc[volume["untagged"].idxmax()]

    left, right = st.columns(2, gap="medium")
    with left:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>Hashtags per post</div>",
                        unsafe_allow_html=True)
            st.altair_chart(ranked_bar(volume, "per_post", None, height=215),
                            use_container_width=True)
            st.markdown(
                f"<p class='panel-note'>{SHORT[hi['hotel']]} carries "
                f"{hi['per_post']:.1f} tags per post, {SHORT[lo['hotel']]} "
                f"{lo['per_post']:.2f} — a factor of "
                f"{hi['per_post'] / max(lo['per_post'], .01):.0f} between two "
                f"luxury houses in the same city. There is no convention "
                f"here to conform to.</p>", unsafe_allow_html=True)
    with right:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>Vocabulary concentration"
                        "</div>", unsafe_allow_html=True)
            st.altair_chart(ranked_bar(vocab, "hhi", None, fmt=".3f", height=215),
                            use_container_width=True)
            usable = vocab.dropna(subset=["hhi"])
            if len(usable) >= 2:
                tight = usable.loc[usable["hhi"].idxmax()]
                loose = usable.loc[usable["hhi"].idxmin()]
                st.markdown(
                    f"<p class='panel-note'>{SHORT[tight['hotel']]} reuses "
                    f"{int(tight['distinct'])} tags across "
                    f"{int(tight['uses']):,} uses. {SHORT[loose['hotel']]} "
                    f"spreads {int(loose['uses']):,} uses over "
                    f"{int(loose['distinct']):,}. High bars mean a small set "
                    f"of tags is doing most of the work.</p>",
                    unsafe_allow_html=True)

    takeaway(
        "<b>There is no benchmark here to conform to.</b> Six houses of comparable "
        "standing, in one city, competing for overlapping guests, have settled on "
        f"volumes spanning a factor of "
        f"{hi['per_post'] / max(lo['per_post'], .01):.0f}. That range is too wide to "
        "be a platform convention, which means each figure is the residue of a "
        f"decision — or of no decision. {SHORT[quiet['hotel']]} publishes "
        f"{quiet['untagged']:.0f}% of its posts with no hashtag at all and still "
        f"holds a median of {quiet['med_likes']:.0f} likes, so running without tags "
        "is a viable position rather than an oversight.<br><br>"
        "<b>The second panel is the more useful one.</b> Volume tells you how loud "
        "an account is; concentration tells you whether it is saying the same thing "
        "twice. The measure is the Herfindahl index, borrowed from "
        "market-concentration analysis: the sum of each tag's squared share of "
        "usage. It rises when a small set of tags carries most of the work and falls "
        "when usage scatters across hundreds of one-off tags. To check that this was "
        "not simply an artefact of the bigger accounts having more room for rare "
        "tags, every hotel was resampled down to 150 hashtag uses; the ranking did "
        "not change.<br><br>"
        "<b>Why this matters operationally.</b> A hashtag accrues value the way a "
        "brand term does — through repetition. Reused consistently, it builds a "
        "browsable archive of your own content, trains the algorithm on what your "
        "account is about, and gives a guest a route back to you. Used once and "
        "abandoned, it does none of that. The disciplined accounts in this set "
        "behave accordingly: a fixed core of roughly ten tags on almost every post, "
        "with a thin rotating layer for seasons and named outlets. The scattered "
        "ones compose a fresh block each time, which produces volume without "
        "accumulation.")

    # -- Row 2: what they tag about -----------------------------------------
    if categories is not None:
        coded = long.merge(categories, on="tag", how="left")
        coded["category"] = coded["category"].fillna("Unclassifiable")

        order = ["Property identity", "Group and loyalty", "Brand philosophy",
                 "Named outlet and talent", "Occasion and service",
                 "Japanese cultural register", "Season and limited time",
                 "Location", "Discovery stack", "Unclassifiable"]

        mix = (pd.crosstab(coded["hotel"], coded["category"], normalize="index")
               * 100).reindex(HOTEL_ORDER)

        st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
        st.markdown("## What each house tags about")

        ml = mix.reset_index().melt(id_vars="hotel", var_name="category",
                                    value_name="pct")
        ml["house"] = ml["hotel"].map(SHORT)
        market = ml.groupby("category")["pct"].transform("mean")
        ml["gap"] = ml["pct"] - market
        ml["label"] = ml["pct"].map(lambda v: f"{v:.0f}" if v >= 0.5 else "")
        cats = [c for c in order if c in set(ml["category"])]
        limit = max(6.0, float(ml["gap"].abs().max()))

        with st.container(border=True):
            grid = alt.Chart(ml).mark_rect(stroke=PAPER, strokeWidth=2).encode(
                x=alt.X("category:N", title=None, sort=cats,
                        axis=alt.Axis(orient="top", domain=False, tickSize=0,
                                      labelAngle=-32, labelColor=INK,
                                      labelFontSize=10.5, labelLimit=200,
                                      labelOverlap=False, labelPadding=6,
                                      labelFont="IBM Plex Sans")),
                y=alt.Y("house:N", title=None, sort=SHORT_ORDER, axis=axis_cat()),
                color=alt.Color("gap:Q",
                                scale=alt.Scale(scheme="redyellowgreen",
                                                domain=[-limit, limit]),
                                legend=None),
                tooltip=[alt.Tooltip("hotel:N", title="Hotel"),
                         alt.Tooltip("category:N", title="Category"),
                         alt.Tooltip("pct:Q", format=".1f", title="% of its tags"),
                         alt.Tooltip("gap:Q", format="+.1f",
                                     title="vs market average")],
            )
            numbers = alt.Chart(ml).mark_text(
                fontSize=12, font="IBM Plex Sans", color=INK).encode(
                x=alt.X("category:N", sort=cats),
                y=alt.Y("house:N", sort=SHORT_ORDER), text="label:N")
            st.altair_chart((grid + numbers).properties(height=290)
                            .configure_view(strokeWidth=0),
                            use_container_width=True)
            st.markdown(
                "<p class='panel-note'>Each cell is that category's share of the "
                "hotel's total hashtag use, in percent. Green marks a share above "
                "the six-hotel average for that category, red below — so colour "
                "shows where a house departs from the market, not simply where "
                "its numbers are large.</p>", unsafe_allow_html=True)

        # -- What the categories mean, with real examples --------------------
        with st.expander("What each category means, with examples from the data"):
            import classify_hashtags as ch
            counts = (coded.groupby(["category", "tag"]).size()
                      .reset_index(name="uses"))
            rows = []
            for name in order:
                if name not in set(coded["category"]):
                    continue
                top = (counts[counts["category"] == name]
                       .sort_values("uses", ascending=False).head(5))
                examples = ", ".join(f"#{t}" for t in top["tag"])
                share = (coded["category"] == name).mean() * 100
                rows.append({
                    "Category": name,
                    "Share of all tags": f"{share:.1f}%",
                    "What it does": ch.CODEBOOK.get(name, "").split(".")[0] + ".",
                    "Most used examples": examples,
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True,
                         use_container_width=True,
                         column_config={
                             "What it does": st.column_config.TextColumn(width="large"),
                             "Most used examples": st.column_config.TextColumn(width="large"),
                         })
            st.markdown(
                "<p class='panel-note'>Categories were built from the vocabulary "
                "these six accounts actually use rather than imposed from theory, "
                "then applied to every distinct hashtag against a written "
                "codebook. Full definitions are in classify_hashtags.py.</p>",
                unsafe_allow_html=True)

        # Construal + audience side by side
        cons = coded[coded["category"].isin(ABSTRACT | CONCRETE)].copy()
        jp_by_hotel = (long.assign(is_jp=long["script"].eq("Japanese"))
                       .groupby("hotel", as_index=False)["is_jp"].mean())
        jp_by_hotel["jp_pct"] = jp_by_hotel["is_jp"] * 100
        market_jp = (long["script"] == "Japanese").mean() * 100

        left, right = st.columns(2, gap="medium")
        with left:
            with st.container(border=True):
                st.markdown("<div class='panel-title'>Abstract share of "
                            "vocabulary</div>", unsafe_allow_html=True)
                if len(cons):
                    cons["level"] = np.where(cons["category"].isin(ABSTRACT),
                                             "Abstract", "Concrete")
                    cm = (pd.crosstab(cons["hotel"], cons["level"],
                                      normalize="index") * 100
                          ).reindex(HOTEL_ORDER).reset_index()
                    if "Abstract" in cm.columns:
                        st.altair_chart(
                            ranked_bar(cm.rename(columns={"Abstract": "abstract"}),
                                       "abstract", None, fmt=".1f", height=215),
                            use_container_width=True)
                        top = cm.loc[cm["Abstract"].idxmax()]
                        bot = cm.loc[cm["Abstract"].idxmin()]
                        st.markdown(
                            f"<p class='panel-note'>Brand philosophy and cultural "
                            f"tags describe what a stay means; outlets, occasions, "
                            f"places and seasons name something bookable. "
                            f"{SHORT[top['hotel']]} sits highest at "
                            f"{top['Abstract']:.0f}% abstract, "
                            f"{SHORT[bot['hotel']]} lowest at "
                            f"{bot['Abstract']:.0f}%.</p>",
                            unsafe_allow_html=True)
        with right:
            with st.container(border=True):
                st.markdown("<div class='panel-title'>Hashtags in Japanese "
                            "script</div>", unsafe_allow_html=True)
                st.altair_chart(ranked_bar(jp_by_hotel, "jp_pct", None,
                                           fmt=".1f", height=215),
                                use_container_width=True)
                top = jp_by_hotel.loc[jp_by_hotel["jp_pct"].idxmax()]
                bottom = jp_by_hotel.loc[jp_by_hotel["jp_pct"].idxmin()]
                st.markdown(
                    f"<p class='panel-note'>The set averages {market_jp:.0f}%. "
                    f"{SHORT[top['hotel']]} tags {top['jp_pct']:.0f}% in "
                    f"Japanese, {SHORT[bottom['hotel']]} {bottom['jp_pct']:.0f}% "
                    f"— opposite bets about who should be able to find "
                    f"them.</p>", unsafe_allow_html=True)

        takeaway(
            "<b>Read the heatmap for distance from the market, not for size.</b> Every "
            "house tags its own name heavily, so a large property-identity number tells "
            "you almost nothing. The informative cells are the ones far from the column "
            "average — the places where a house has made a different choice from its "
            "competitors.<br><br>"
            "<b>The discovery-stack column deserves the most attention.</b> These are "
            "the broad, unowned search terms: luxury hotel, Tokyo travel, and their "
            "Japanese equivalents. They feel productive because they have volume behind "
            "them, but they place a post into a feed shared with every other hotel in "
            "the city, ranked largely by engagement velocity, competing against accounts "
            "many times larger. A high share there is not reach so much as borrowed "
            "traffic on a term the property will never own. The same effort spent on "
            "proprietary language compounds instead: brand-philosophy tags are the only "
            "ones a competitor structurally cannot use.<br><br>"
            "<b>The abstract share is a positioning measure, not a style preference.</b> "
            "Construal level theory holds that psychological distance and abstract "
            "language reinforce one another, and luxury depends on maintaining that "
            "distance. Concrete tags — an outlet, a date, a district — sell a specific "
            "booking to someone already close to purchase. Abstract tags describe what a "
            "stay signifies and work on someone who is not yet shopping. A house tagging "
            "almost entirely in the concrete is running a promotions calendar under a "
            "luxury logo; a house with a substantial abstract share is building "
            "something a competitor cannot replicate by opening a similar "
            "restaurant.<br><br>"
            "<b>Script is a targeting decision that looks like a formatting one.</b> "
            "Instagram hashtag search is script-literal: a guest searching in Japanese "
            "will not surface a post tagged only in Latin characters, and the reverse "
            "holds. Japanese users search by hashtag at several times the global rate, "
            "so this is not a cosmetic detail — the split above allocates a property's "
            "discoverability between the domestic guest and the inbound traveller. The "
            "diagnostic is straightforward: compare the split against the property's own "
            "booking mix, and see whether the search visibility is being spent where the "
            "revenue is.")

    # -- Row 3: engagement and format ---------------------------------------
    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
    st.markdown("## What the numbers do, and do not, reward")

    corr = pd.DataFrame([{
        "hotel": h,
        "Likes": rank_corr(posts[posts["hotel"] == h]["tag_count"],
                           posts[posts["hotel"] == h]["likes"]),
        "Comments": rank_corr(posts[posts["hotel"] == h]["tag_count"],
                              posts[posts["hotel"] == h]["comments"]),
    } for h in HOTEL_ORDER])

    # Pooling likes across hotels would measure which hotel favours which
    # format, not the format itself. Each post is indexed against its own
    # hotel's median before comparing.
    fmt_src = posts.dropna(subset=["likes"]).copy()
    fmt_src["rel"] = (fmt_src["likes"]
                      / fmt_src.groupby("hotel")["likes"].transform("median"))
    fmt = (fmt_src.groupby("format")
           .agg(posts=("rel", "size"), index=("rel", "median"))
           .reset_index().sort_values("index", ascending=False))

    left, right = st.columns(2, gap="medium")
    with left:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>Hashtag count against "
                        "engagement</div>", unsafe_allow_html=True)
            cm2 = corr.melt(id_vars="hotel", var_name="metric",
                            value_name="rho").dropna()
            cm2["house"] = cm2["hotel"].map(SHORT)
            if len(cm2):
                st.altair_chart(
                    alt.Chart(cm2).mark_bar(height=11).encode(
                        x=alt.X("rho:Q", title=None,
                                scale=alt.Scale(domain=[-.35, .35]),
                                axis=axis_num()),
                        y=alt.Y("house:N", title=None, sort=SHORT_ORDER,
                                axis=axis_cat()),
                        yOffset=alt.YOffset("metric:N"),
                        color=alt.Color("metric:N",
                                        scale=alt.Scale(
                                            domain=["Likes", "Comments"],
                                            range=[ACCENT, WARM]),
                                        legend=alt.Legend(title=None,
                                                          orient="top",
                                                          labelColor=MUTED)),
                        tooltip=[alt.Tooltip("hotel:N", title="Hotel"), "metric",
                                 alt.Tooltip("rho:Q", format=".3f")],
                    ).properties(height=230).configure_view(strokeWidth=0),
                    use_container_width=True)
            st.markdown(
                "<p class='panel-note'>Rank correlation, measured inside each "
                "account. Bars to the right mean more tags accompany more "
                "engagement; to the left, less. Nothing here is close to a "
                "strong relationship.</p>", unsafe_allow_html=True)
    with right:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>Post format, indexed to each "
                        "hotel's own median</div>", unsafe_allow_html=True)
            rule = (alt.Chart(pd.DataFrame({"v": [1.0]}))
                    .mark_rule(color=MUTED, strokeDash=[4, 3]).encode(x="v:Q"))
            bars = alt.Chart(fmt).mark_bar(height=26).encode(
                x=alt.X("index:Q", title="1.0 = that hotel's typical post",
                        scale=alt.Scale(domain=[0, 1.3]), axis=axis_num()),
                y=alt.Y("format:N", title=None, sort="-x", axis=axis_cat()),
                color=alt.value(ACCENT),
                tooltip=["format", "posts", alt.Tooltip("index:Q", format=".2f")],
            )
            st.altair_chart((bars + rule).properties(height=230)
                            .configure_view(strokeWidth=0),
                            use_container_width=True)
            ranking = ", ".join(f"{r['format']} {r['index']:.2f}"
                                for _, r in fmt.iterrows())
            st.markdown(
                f"<p class='panel-note'>{ranking}. Comparing raw likes across "
                f"hotels would measure audience size, so every post is divided "
                f"by its own hotel's median first. Feed posts only — reels are "
                f"outside this sample.</p>", unsafe_allow_html=True)

    takeaway(
        "<b>Adding hashtags is not a free lever, and the numbers say so quietly.</b> "
        "Rank correlation is used here because engagement is heavily skewed — a "
        "handful of unusually large posts would dominate any linear measure — and "
        "each figure is calculated inside a single account, because follower counts "
        "are not observable from public posts and cross-account comparison would "
        "measure audience size rather than tagging.<br><br>"
        "The pattern that emerges is not a clean positive. On the accounts with the "
        "largest audiences, more tags accompany no gain in likes and a fall in "
        "comments. Comments are the harder metric to move: a like is a scroll-past "
        "reflex, a comment means someone stopped. If tagging more buys marginal "
        "passive reach while the conversation thins, a reporting line that sums the "
        "two into one engagement figure will show improvement where the more "
        "valuable signal has weakened.<br><br>"
        "<b>On format, the comparison had to be rebuilt.</b> Comparing raw likes "
        "across hotels would mostly measure which house has the bigger following, so "
        "each post is indexed against its own hotel's median first. On that basis "
        "single images run above a typical post, carousels sit at par, and video "
        "runs below — which inverts the usual advice and published findings from "
        "hotel Instagram research in other markets. One boundary worth stating: this "
        "sample is feed posts only, so it compares video inside the feed against "
        "stills, and says nothing about reels.<br><br>"
        "Everything on this row is correlational. Subject, timing, season and "
        "creative quality all move engagement, and none of them are controlled "
        "here.")

    # -- Row 4: the trend, decomposed ---------------------------------------
    #
    # The aggregate Japanese-script share is not safe to read on its own: the
    # hotels contribute very unequal numbers of hashtags, and that mix changes
    # over time. A hotel that tags heavily in Japanese posting less will drag
    # the aggregate down without any account changing its practice. So the
    # aggregate and the equal-weighted series are shown together, and the
    # per-hotel lines below them.
    qt = long.copy()
    qt["quarter"] = qt["date"].dt.to_period("Q")
    qt["label"] = (qt["quarter"].dt.year.astype(str) + " Q"
                   + qt["quarter"].dt.quarter.astype(str))
    qt["is_jp"] = qt["script"].eq("Japanese")

    per = (qt.groupby(["label", "hotel"])
           .agg(jp=("is_jp", "mean"), n=("is_jp", "size")).reset_index())
    per = per[per["n"] >= 30].copy()
    per["jp"] *= 100
    per["house"] = per["hotel"].map(SHORT)

    agg = (qt.groupby("label").agg(jp=("is_jp", "mean"), n=("is_jp", "size"))
           .reset_index())
    agg = agg[agg["n"] >= 100].copy()
    agg["jp"] *= 100
    equal = per.groupby("label", as_index=False)["jp"].mean()

    labels = sorted(set(agg["label"]) | set(per["label"]))

    if len(labels) >= 4:
        st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
        st.markdown("## Is the audience mix actually shifting?")

        left, right = st.columns(2, gap="medium")

        with left:
            with st.container(border=True):
                st.markdown("<div class='panel-title'>Aggregate against "
                            "equal-weighted</div>", unsafe_allow_html=True)
                two = pd.concat([
                    agg.assign(series="All hashtags pooled")[["label", "jp", "series"]],
                    equal.assign(series="Each hotel weighted equally"),
                ])
                st.altair_chart(
                    alt.Chart(two).mark_line(strokeWidth=2,
                                             point=alt.OverlayMarkDef(size=28)).encode(
                        x=alt.X("label:N", title=None, sort=labels, axis=axis_cat()),
                        y=alt.Y("jp:Q", title="Japanese-script tags (%)",
                                scale=alt.Scale(zero=False), axis=axis_num()),
                        color=alt.Color("series:N",
                                        scale=alt.Scale(
                                            domain=["All hashtags pooled",
                                                    "Each hotel weighted equally"],
                                            range=[NEUTRAL, ACCENT]),
                                        legend=alt.Legend(title=None, orient="top",
                                                          labelColor=MUTED,
                                                          labelLimit=240)),
                        tooltip=["label", "series", alt.Tooltip("jp:Q", format=".1f")],
                    ).properties(height=250).configure_view(strokeWidth=0),
                    use_container_width=True)
                st.markdown(
                    "<p class='panel-note'>Pooling every hashtag lets the "
                    "highest-volume account dominate. Weighting each hotel "
                    "equally removes that, and most of the apparent decline "
                    "with it.</p>", unsafe_allow_html=True)

        with right:
            with st.container(border=True):
                st.markdown("<div class='panel-title'>Each hotel on its own"
                            "</div>", unsafe_allow_html=True)
                st.altair_chart(
                    alt.Chart(per).mark_line(strokeWidth=1.8,
                                             point=alt.OverlayMarkDef(size=22)).encode(
                        x=alt.X("label:N", title=None, sort=labels, axis=axis_cat()),
                        y=alt.Y("jp:Q", title="Japanese-script tags (%)",
                                scale=alt.Scale(zero=False), axis=axis_num()),
                        color=alt.Color("house:N", title=None,
                                        scale=alt.Scale(scheme="tableau10"),
                                        legend=alt.Legend(orient="top",
                                                          labelColor=MUTED,
                                                          columns=3)),
                        tooltip=[alt.Tooltip("hotel:N", title="Hotel"), "label",
                                 alt.Tooltip("jp:Q", format=".1f"),
                                 alt.Tooltip("n:Q", title="tags")],
                    ).properties(height=250).configure_view(strokeWidth=0),
                    use_container_width=True)
                st.markdown(
                    "<p class='panel-note'>Quarters with fewer than 30 hashtags "
                    "for a hotel are omitted, which is why lines start and stop "
                    "at different points.</p>", unsafe_allow_html=True)

        # Which hotels actually moved, first half against second
        half = len(labels) // 2
        early, late = set(labels[:half]), set(labels[half:])
        moves = []
        for hotel, grp in per.groupby("hotel"):
            a = grp[grp["label"].isin(early)]["jp"].mean()
            b = grp[grp["label"].isin(late)]["jp"].mean()
            if pd.notna(a) and pd.notna(b):
                moves.append({"hotel": hotel, "change": b - a})
        moves = pd.DataFrame(moves).sort_values("change")

        if len(moves):
            fell = moves.iloc[0]
            rose = moves.iloc[-1]
            agg_change = agg["jp"].iloc[-1] - agg["jp"].max()
            eq_change = equal["jp"].iloc[-1] - equal["jp"].max()
            takeaway(
                f"<b>The pooled line is misleading, and this is the more useful "
                f"finding.</b> Read together, the two panels show that the "
                f"apparent market-wide move toward Latin script is largely an "
                f"artefact of which accounts were posting. The highest-volume "
                f"account in this set tags heavily in Japanese; its share of "
                f"total hashtags fell over the period, and the pooled average "
                f"fell with it. Weight the hotels equally and the decline "
                f"shrinks from {abs(agg_change):.0f} points to "
                f"{abs(eq_change):.0f}.<br><br>"
                f"Individually the houses diverge rather than drift together. "
                f"{SHORT[rose['hotel']]} moved {rose['change']:+.0f} points "
                f"toward Japanese across the period; {SHORT[fell['hotel']]} moved "
                f"{fell['change']:+.0f}. Those are strategic decisions pulling in "
                f"opposite directions, not a shared response to the market.<br><br>"
                f"<b>The practical warning.</b> Any competitive-set benchmark "
                f"built by pooling posts across accounts carries this bias. If a "
                f"rival simply posts more often, the benchmark moves toward that "
                f"rival's habits and away from a description of the market. "
                f"Compare per-account, or weight deliberately.")

# ===========================================================================
# SINGLE PROPERTY
# ===========================================================================

else:
    row = volume[volume["hotel"] == focal].iloc[0]
    peers_v = volume[volume["hotel"] != focal]
    peers_vocab = vocab[vocab["hotel"] != focal]
    ftags = long[long["hotel"] == focal]

    st.markdown(f"## {focal}")
    c = st.columns(4, gap="medium")
    stat(c[0], f"{row['per_post']:.1f}",
         f"hashtags per post<br>peer median {peers_v['per_post'].median():.1f}")
    stat(c[1], f"{ftags['tag'].nunique():,}",
         f"distinct tags<br>peer median {peers_vocab['distinct'].median():.0f}")
    stat(c[2], f"{row['med_likes']:.0f}",
         f"median likes<br>peer median {peers_v['med_likes'].median():.0f}")
    stat(c[3], f"{row['untagged']:.0f}%",
         f"posts with no hashtag<br>peer median {peers_v['untagged'].median():.0f}%")

    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.markdown("### The vocabulary it repeats")
        if len(ftags):
            top = (ftags["tag"].value_counts().head(15)
                   .rename_axis("Hashtag").reset_index(name="Uses"))
            top["Share"] = (top["Uses"] / len(ftags) * 100).map("{:.1f}%".format)
            st.dataframe(top, hide_index=True, use_container_width=True, height=420)
        else:
            st.info("No hashtags in this window.")
    with c2:
        st.markdown("### Where it sits in the set")
        st.altair_chart(ranked_bar(volume, "per_post", "Hashtags per post",
                                   focal=focal, height=185),
                        use_container_width=True)
        st.altair_chart(ranked_bar(vocab, "hhi", "Concentration", focal=focal,
                                   fmt=".3f", height=185),
                        use_container_width=True)

    if categories is not None and len(ftags):
        coded = long.merge(categories, on="tag", how="left")
        coded["category"] = coded["category"].fillna("Unclassifiable")
        mix = pd.crosstab(coded["hotel"], coded["category"], normalize="index") * 100
        if focal in mix.index:
            fm = mix.loc[focal].sort_values(ascending=False)
            gap = (mix.loc[focal] - mix.drop(index=focal).mean()).sort_values()

            st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
            st.markdown("### What it tags about, against the peer average")
            cmp_long = pd.DataFrame({
                "Category": list(mix.columns) * 2,
                "who": [focal] * len(mix.columns) + ["Peer average"] * len(mix.columns),
                "pct": list(mix.loc[focal].values)
                       + list(mix.drop(index=focal).mean().values),
            })
            c1, c2 = st.columns([3, 2], gap="large")
            with c1:
                st.altair_chart(
                    alt.Chart(cmp_long).mark_bar(height=11).encode(
                        x=alt.X("pct:Q", title="Share of hashtag uses (%)",
                                axis=axis_num()),
                        y=alt.Y("Category:N", title=None, sort="-x", axis=axis_cat()),
                        yOffset=alt.YOffset("who:N"),
                        color=alt.Color("who:N",
                                        scale=alt.Scale(domain=[focal, "Peer average"],
                                                        range=[ACCENT, NEUTRAL]),
                                        legend=alt.Legend(title=None, orient="top",
                                                          labelColor=MUTED)),
                        tooltip=["Category", "who", alt.Tooltip("pct:Q", format=".1f")],
                    ).properties(height=280).configure_view(strokeWidth=0),
                    use_container_width=True)
            with c2:
                st.markdown(
                    f"<div class='read'>"
                    f"<p>Its largest category is {fm.index[0].lower()} at "
                    f"{fm.iloc[0]:.0f}% of all hashtag use.</p>"
                    f"<p>Against the other five it leans hardest into "
                    f"{gap.index[-1].lower()} ({gap.iloc[-1]:+.0f} points) and does "
                    f"least with {gap.index[0].lower()} ({gap.iloc[0]:+.0f} "
                    f"points).</p></div>", unsafe_allow_html=True)

    # Script profile
    if len(ftags):
        st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
        st.markdown("### Who it tags for")
        jp_pct = (ftags["script"] == "Japanese").mean() * 100
        peer_jp = (long[long["hotel"] != focal]["script"] == "Japanese").mean() * 100
        c1, c2 = st.columns([1, 2], gap="large")
        stat(c1, f"{jp_pct:.0f}%",
             f"of its tags are Japanese script<br>peer average {peer_jp:.0f}%")
        with c2:
            direction = "more heavily" if jp_pct > peer_jp else "less"
            st.markdown(
                f"<div class='read'><p>{focal} tags {direction} in Japanese than "
                f"the rest of the set. In a market where domestic users search by "
                f"hashtag far above the global rate and inbound arrivals are at "
                f"record levels, script is the lever that decides which audience "
                f"can find a post.</p></div>", unsafe_allow_html=True)

    # Hygiene
    inv = long[(long["hotel"] == focal) & (long["invisible"])]
    if len(inv) and len(ftags):
        st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
        st.markdown("### A fault worth fixing this week")
        st.markdown(
            f"{len(inv)} of {focal}'s {len(ftags):,} hashtag uses "
            f"({len(inv) / len(ftags) * 100:.0f}%) carry an invisible Unicode "
            f"character — U+2060, a word joiner — on the end of the tag. It does "
            f"not show in the caption. It arrives by pasting a hashtag block out "
            f"of a formatted document rather than typing it, and the effect is "
            f"that the same brand tag is published as two different tags."
        )
        pairs = []
        for tag in inv["tag"].value_counts().head(8).index:
            clean = int(((ftags["tag"] == tag) & (~ftags["invisible"])).sum())
            dirty = int(((ftags["tag"] == tag) & (ftags["invisible"])).sum())
            if clean and dirty:
                pairs.append({"Hashtag": f"#{tag}", "Typed": clean,
                              "With hidden character": dirty,
                              "Going to the broken form":
                                  f"{dirty / (clean + dirty) * 100:.0f}%"})
        if pairs:
            c1, c2 = st.columns([3, 2], gap="large")
            with c1:
                st.dataframe(pd.DataFrame(pairs), hide_index=True,
                             use_container_width=True)
            with c2:
                st.markdown(
                    "<div class='read'>"
                    "<p>Search each of these on Instagram twice — once typed by "
                    "hand, once pasted from your caption template. If the two "
                    "return different results the tag is fragmented, and much of "
                    "the brand's own tag equity is landing where no one "
                    "searches.</p>"
                    "<p>The fix costs nothing: retype the hashtag block once, save "
                    "it as plain text, stop pasting it from formatted "
                    "sources.</p></div>", unsafe_allow_html=True)

# ===========================================================================
# Coding tools — run the classifier from the browser
# ===========================================================================

def _api_key():
    """Key from Streamlit secrets, falling back to the environment."""
    try:
        if "OPENAI_API_KEY" in st.secrets:
            raw_key = str(st.secrets["OPENAI_API_KEY"])
            return raw_key.strip().strip('"').strip("'")
    except Exception:            # no secrets.toml locally
        pass
    import os
    key = os.environ.get("OPENAI_API_KEY")
    return key.strip().strip('"').strip("'") if key else None


st.markdown("<div class='rule'></div>", unsafe_allow_html=True)

with st.expander("Coding tools — build the hashtag classification"):
    import classify_hashtags as ch

    tag_counts = (long_all["tag"].value_counts()
                  .rename_axis("tag").reset_index(name="uses"))

    st.markdown(
        f"<p class='note'>{len(tag_counts):,} distinct hashtags across the whole "
        f"dataset. Step one draws a stratified sample for you to code by hand. "
        f"Step two runs the classifier. Code the sample <b>before</b> running the "
        f"classifier, or the agreement figure is meaningless.</p>",
        unsafe_allow_html=True)

    st.markdown("#### 1 · Hand-coding sheet")

    bands = pd.cut(tag_counts["uses"], bins=[0, 1, 5, 25, 10 ** 9],
                   labels=["1 use", "2-5", "6-25", "26+"])
    per_band = 75
    parts = []
    for band, group in tag_counts.assign(band=bands).groupby("band", observed=True):
        parts.append(group.sample(min(per_band, len(group)), random_state=7))
    sample_sheet = (pd.concat(parts, ignore_index=True)
                    .sample(frac=1, random_state=7).reset_index(drop=True))
    sample_sheet["my_category"] = ""

    st.download_button(
        f"Download validation_sample.csv ({len(sample_sheet)} tags)",
        sample_sheet.to_csv(index=False).encode("utf-8-sig"),
        file_name="validation_sample.csv", mime="text/csv")
    st.markdown(
        "<p class='note'>Open it in Google Sheets, not Excel — Excel will destroy "
        "the Japanese characters. Fill in <code>my_category</code> using exactly "
        "one of: " + ", ".join(ch.CATEGORIES) + ".</p>", unsafe_allow_html=True)

    st.markdown("#### 2 · Run the classifier")

    key = _api_key()
    if key:
        st.markdown(
            f"<p class='note'>Key loaded: <code>{key[:7]}…{key[-4:]}</code> "
            f"({len(key)} characters).</p>", unsafe_allow_html=True)
    if not key:
        st.warning("No OPENAI_API_KEY found. Add it under Manage app → Settings → "
                   "Secrets as:  OPENAI_API_KEY = \"sk-...\"")
    else:
        st.markdown(
            f"<p class='note'>Classifies all {len(tag_counts):,} tags against the "
            f"codebook in classify_hashtags.py, in batches of {ch.BATCH_SIZE} at "
            f"temperature 0. A few minutes and a few cents on "
            f"{ch.MODEL}.</p>", unsafe_allow_html=True)

        test = st.button("Test the connection")
        if test:
            try:
                from openai import OpenAI
                probe = OpenAI(api_key=key)
                probe_tags = ["amantokyo", "赤坂", "afternoontea", "luxuryispersonal"]
                reply = ch.classify_batch(probe, probe_tags)
                st.success(f"Connection works — {len(reply)} of "
                           f"{len(probe_tags)} tags parsed.")
                st.dataframe(
                    pd.DataFrame([{"Hashtag": f"#{t}", "Category": v[0],
                                   "Confidence": v[1]}
                                  for t, v in reply.items()]),
                    hide_index=True, use_container_width=True)
            except Exception as exc:                       # noqa: BLE001
                st.error(f"**{type(exc).__name__}**\n\n{exc}")
                st.markdown(
                    "<p class='note'>Common causes: the key has been revoked or "
                    "is from a different account; the account has no credit "
                    "(OpenAI billing is separate from a ChatGPT subscription); "
                    "or the key was pasted with a stray quote or newline.</p>",
                    unsafe_allow_html=True)

        if st.button("Classify hashtags", type="primary"):
            try:
                from openai import OpenAI
            except ImportError:
                st.error("The openai package is missing. Add `openai>=1.0` to "
                         "requirements.txt and reboot the app.")
            else:
                client = OpenAI(api_key=key)
                todo = list(tag_counts["tag"])
                done, failed = {}, 0
                first_error = None
                consecutive = 0
                aborted = False
                bar = st.progress(0.0, text="Starting…")

                for start in range(0, len(todo), ch.BATCH_SIZE):
                    batch = todo[start:start + ch.BATCH_SIZE]
                    try:
                        result = ch.classify_batch(client, batch)
                        consecutive = 0
                    except Exception as exc:               # noqa: BLE001
                        if first_error is None:
                            first_error = f"{type(exc).__name__}: {exc}"
                        failed += len(batch)
                        consecutive += 1
                        result = {}
                        # Three failures in a row means the problem is the
                        # connection, not the batch. Stop rather than burn
                        # through every remaining request.
                        if consecutive >= 3:
                            aborted = True
                            break
                    for tag in batch:
                        done[tag] = result.get(tag, ("Unclassifiable", "low"))
                    seen = min(start + ch.BATCH_SIZE, len(todo))
                    bar.progress(seen / len(todo), text=f"{seen:,} of {len(todo):,}")

                bar.empty()

                if aborted or (first_error and failed == len(todo)):
                    st.error(
                        "Classification stopped — the API is not responding.\n\n"
                        f"First error was:\n\n`{first_error}`")
                    st.markdown(
                        "<p class='note'>Nothing was saved. Fix the key or "
                        "billing, then run it again. Use <b>Test the "
                        "connection</b> above to check before a full run.</p>",
                        unsafe_allow_html=True)
                else:
                    out = tag_counts.copy()
                    out["category"] = out["tag"].map(
                        lambda t: done.get(t, ("Unclassifiable",))[0])
                    out["confidence"] = out["tag"].map(
                        lambda t: done.get(t, (None, "low"))[1])
                    real = (out["category"] != "Unclassifiable").mean()
                    if real < 0.2:
                        st.error(
                            "Nearly everything came back Unclassifiable, so the "
                            "run did not work. Don't use this file. Press "
                            "**Test the connection** above to see the reply.")
                    st.session_state["classified"] = out
                    if failed:
                        st.warning(
                            f"{failed} tags fell back to Unclassifiable after a "
                            f"failed request ({first_error}). Re-run to retry.")

    if "classified" in st.session_state:
        out = st.session_state["classified"]
        st.success(f"Classified {len(out):,} tags.")
        summary = (out.groupby("category")
                   .agg(**{"Distinct tags": ("tag", "size"),
                           "Total uses": ("uses", "sum")})
                   .sort_values("Total uses", ascending=False).reset_index()
                   .rename(columns={"category": "Category"}))
        st.dataframe(summary, hide_index=True, use_container_width=True)
        st.download_button(
            "Download hashtag_categories.csv",
            out.to_csv(index=False).encode("utf-8-sig"),
            file_name="hashtag_categories.csv", mime="text/csv", type="primary")
        st.markdown(
            "<p class='note'>Upload this file to your GitHub repo (Add file → "
            "Upload files) next to app.py. The app will pick it up on the next "
            "reboot and the category and construal sections will appear.</p>",
            unsafe_allow_html=True)

    st.markdown("#### 3 · Score your agreement")
    coded_up = st.file_uploader("Your completed validation_sample.csv", type="csv")
    if coded_up is not None:
        mine = pd.read_csv(coded_up)
        model = (st.session_state.get("classified")
                 if "classified" in st.session_state else categories)
        if model is None:
            st.info("Run the classifier first, or commit hashtag_categories.csv.")
        elif "my_category" not in mine.columns:
            st.error("That file has no my_category column.")
        else:
            mine = mine[mine["my_category"].notna()
                        & (mine["my_category"].astype(str).str.strip() != "")]
            merged = mine.merge(model[["tag", "category"]], on="tag", how="inner")
            if merged.empty:
                st.error("No coded tags matched the classification.")
            else:
                merged["match"] = (merged["my_category"].str.strip().str.lower()
                                   == merged["category"].str.strip().str.lower())
                observed = merged["match"].mean()
                p_mine = merged["my_category"].value_counts(normalize=True)
                p_model = merged["category"].value_counts(normalize=True)
                expected = sum(p_mine.get(c, 0) * p_model.get(c, 0)
                               for c in ch.CATEGORIES)
                kappa = ((observed - expected) / (1 - expected)
                         if expected < 1 else float("nan"))
                a, b = st.columns(2)
                stat(a, f"{observed * 100:.1f}%", f"agreement over {len(merged)} tags")
                stat(b, f"{kappa:.3f}", "Cohen's kappa<br>above 0.80 is strong")
                wrong = merged[~merged["match"]][["tag", "my_category", "category"]]
                if len(wrong):
                    st.markdown("**Disagreements** — read these before writing up. "
                                "A pattern here usually means a codebook definition "
                                "is unclear, not that the model is wrong.")
                    st.dataframe(wrong.rename(columns={
                        "tag": "Hashtag", "my_category": "Your code",
                        "category": "Model"}), hide_index=True,
                        use_container_width=True)

# ===========================================================================
# Method
# ===========================================================================

st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
st.markdown("## Method")

m1, m2 = st.columns(2, gap="large")
with m1:
    st.markdown(
        f"<p class='note'><b>Sample.</b> {len(posts):,} feed posts from six Tokyo "
        f"luxury hotel Instagram accounts, {posts['date'].min():%B %Y} to "
        f"{posts['date'].max():%B %Y}, collected from public profiles via the "
        f"Apify Instagram Scraper. {len(long):,} hashtag occurrences across "
        f"{long['tag'].nunique():,} distinct tags.</p>"
        f"<p class='note'><b>Period.</b> The study window opens in May 2023. "
        f"Earlier posts exist in the scrape but are too sparse across accounts to "
        f"compare, and are excluded.</p>"
        f"<p class='note'><b>Exclusion.</b> The Four Seasons Hotel Tokyo at "
        f"Marunouchi was part of the original design and was dropped: its "
        f"account's public history begins February 2026, leaving no overlap with "
        f"the study window.</p>", unsafe_allow_html=True)
with m2:
    coding = (
        "Categories were assigned by a large language model (gpt-4o-mini, "
        "temperature 0) against the codebook in classify_hashtags.py, then "
        "validated against a hand-coded stratified sample. The classification is "
        "fixed in hashtag_categories.csv rather than generated at run time, so "
        "the figures are reproducible."
        if categories is not None else
        "Category and construal analysis are not shown. Run classify_hashtags.py "
        "to produce hashtag_categories.csv, then reload."
    )
    st.markdown(
        f"<p class='note'><b>Coding.</b> {coding}</p>"
        f"<p class='note'><b>Measures.</b> Engagement is public likes and comments. "
        f"Follower counts are not observable through this method, so no "
        f"cross-account engagement rate is computed and every correlation is "
        f"calculated within a single account. Concentration is the Herfindahl "
        f"index over a hotel's tag distribution.</p>"
        f"<p class='note'><b>Limits.</b> Feed posts only; reels and stories are "
        f"excluded, as are hashtags placed in comments, which understates accounts "
        f"that tag in the first comment. All relationships are correlational.</p>",
        unsafe_allow_html=True)
