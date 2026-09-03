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
DATA_FILE = "dataset_instagram-scraper-task-3_2026-09-02_10-51-38-883.csv"
CATEGORY_FILE = HERE / "hashtag_categories.csv"

# Two sample windows. The full window maximises observations; the matched
# window starts where the shortest-lived account in the original seven-hotel
# design begins, and is reported as a robustness check.
WINDOWS = {
    "Full window — May 2023 onward": "2023-05-01",
    "Matched window — February 2026 onward": "2026-02-18",
}

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
  .paper {{ background:#F4F1E9; border-radius:2px; padding:.85rem 1rem;
            font-size:.83rem; color:#4A483F; line-height:1.55; margin-top:.9rem; }}
  .paper b {{ color:{INK}; font-weight:600; }}
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
    for pattern in ("dataset_instagram-scraper*.csv", "*.csv"):
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
    return alt.Axis(domain=False, tickSize=0, labelColor=INK, labelFontSize=12,
                    labelPadding=8, labelFont="IBM Plex Sans")


def ranked_bar(data, value, value_title, focal=None, fmt=".2f", height=210,
               sort_order=None):
    d = data.dropna(subset=[value]).copy()
    d["_focal"] = d["hotel"] == focal if focal else False
    colour = (alt.condition(alt.datum._focal, alt.value(ACCENT), alt.value(NEUTRAL))
              if focal else alt.value(ACCENT))
    return (alt.Chart(d).mark_bar(height=16).encode(
        x=alt.X(f"{value}:Q", title=value_title, axis=axis_num()),
        y=alt.Y("hotel:N", title=None, sort=sort_order or HOTEL_ORDER,
                axis=axis_cat()),
        color=colour,
        tooltip=["hotel", alt.Tooltip(f"{value}:Q", title=value_title, format=fmt)],
    ).properties(height=height).configure_view(strokeWidth=0))


def stat(container, value, label, small=False):
    cls = "stat-sm" if small else "stat"
    container.markdown(f"<div class='{cls}'>{value}</div>"
                       f"<div class='stat-label'>{label}</div>",
                       unsafe_allow_html=True)


def paper_note(text):
    st.markdown(f"<div class='paper'>{text}</div>", unsafe_allow_html=True)


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
    window_label = st.selectbox("Sample window", list(WINDOWS), index=0)
    view = st.radio("View", ["Competitive set", "Single property"],
                    horizontal=True)

cutoff = WINDOWS[window_label]
posts = posts_all[posts_all["date"] >= cutoff].copy()
long = long_all[long_all["date"] >= cutoff].copy()

focal = None
if view == "Single property":
    focal = st.selectbox("Property", HOTEL_ORDER,
                         index=HOTEL_ORDER.index("Park Hyatt Tokyo"))

# A Herfindahl index over a handful of observations is not interpretable, and
# neither is a rank correlation. MIN_USES gates the concentration figures.
MIN_USES = 40

if len(posts) < 600:
    st.warning(
        f"This window holds {len(posts):,} posts and {len(long):,} hashtag uses. "
        "It is the robustness check, not the primary analysis — hotels with very "
        "few tagged posts are flagged below and some statistics are withheld. "
        "Switch to the full window for the headline figures."
    )

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

    st.markdown("## The set at a glance")
    cols = st.columns(6, gap="small")
    for col, hotel in zip(cols, HOTEL_ORDER):
        r = volume[volume["hotel"] == hotel].iloc[0]
        col.markdown(
            f"<div class='stat-label' style='color:{INK};font-weight:600;"
            f"min-height:2.6em'>{SHORT[hotel]}</div>"
            f"<div class='stat-sm'>{r['per_post']:.1f}</div>"
            f"<div class='stat-label'>tags per post<br>{int(r['posts'])} posts"
            f"<br>{r['med_likes']:.0f} median likes</div>",
            unsafe_allow_html=True)

    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)

    # -- Volume -------------------------------------------------------------
    st.markdown("## No shared convention on how much to tag")
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.altair_chart(ranked_bar(volume, "per_post", "Hashtags per post"),
                        use_container_width=True)
    with c2:
        lo = volume.loc[volume["per_post"].idxmin()]
        hi = volume.loc[volume["per_post"].idxmax()]
        q = volume.loc[volume["untagged"].idxmax()]
        st.markdown(
            f"<div class='read'>"
            f"<p>Practice spans a factor of {hi['per_post'] / max(lo['per_post'], .01):.0f}. "
            f"{SHORT[lo['hotel']]} averages {lo['per_post']:.2f} hashtags per post; "
            f"{SHORT[hi['hotel']]} averages {hi['per_post']:.1f}.</p>"
            f"<p>{SHORT[q['hotel']]} publishes {q['untagged']:.0f}% of its posts with "
            f"no hashtag at all and still holds a median of {q['med_likes']:.0f} likes. "
            f"Abstention is a live strategy here, not an oversight.</p></div>",
            unsafe_allow_html=True)
        paper_note("<b>Paper:</b> descriptive answer to RQ1. Establishes hashtag "
                   "volume as a discretionary brand decision rather than a "
                   "platform norm — the precondition for reading it as curated "
                   "visibility rather than routine practice.")

    # -- Vocabulary ---------------------------------------------------------
    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
    st.markdown("## Discipline measures more than volume does")
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.altair_chart(ranked_bar(vocab, "hhi", "Concentration (Herfindahl)",
                                   fmt=".3f"), use_container_width=True)
        show = vocab.rename(columns={"hotel": "Hotel", "distinct": "Distinct tags",
                                     "hhi": "Concentration", "uses": "Hashtag uses"})
        show["Concentration"] = show["Concentration"].map(
            lambda v: "—" if pd.isna(v) else f"{v:.3f}")
        st.dataframe(show, hide_index=True, use_container_width=True)
        if vocab["hhi"].isna().any():
            st.markdown(
                f"<p class='note'>Concentration is withheld (—) for hotels with "
                f"fewer than {MIN_USES} hashtag uses in this window; the index is "
                f"not meaningful on a handful of observations.</p>",
                unsafe_allow_html=True)
    with c2:
        usable = vocab.dropna(subset=["hhi"])
        if len(usable) < 2:
            st.info(f"Too few hashtag uses in this window to compare "
                    f"concentration. Hotels need at least {MIN_USES} uses.")
            tight = loose = None
        else:
            tight = usable.loc[usable["hhi"].idxmax()]
            loose = usable.loc[usable["hhi"].idxmin()]
        if tight is not None:
            st.markdown(
                f"<div class='read'>"
                f"<p>Counting tags measures how loud an account is. Concentration "
                f"measures whether the same language returns post after post — which "
                f"is what accumulates into a searchable brand vocabulary.</p>"
                f"<p>{SHORT[tight['hotel']]} draws on {int(tight['distinct'])} distinct "
                f"tags across {int(tight['uses']):,} uses. {SHORT[loose['hotel']]} "
                f"spreads {int(loose['uses']):,} uses across {int(loose['distinct']):,}. "
                f"Both are choices; only one compounds.</p></div>",
                unsafe_allow_html=True)
        paper_note("<b>Paper:</b> operationalises curated visibility as a "
                   "measurable property. Concentration separates deliberate "
                   "curation from sheer volume — the gap the review identifies "
                   "in studies that count hashtags alone.")

    # -- Categories ---------------------------------------------------------
    if categories is not None:
        coded = long.merge(categories, on="tag", how="left")
        coded["category"] = coded["category"].fillna("Unclassifiable")

        order = ["Property identity", "Group and loyalty", "Brand philosophy",
                 "Named outlet and talent", "Occasion and service",
                 "Japanese cultural register", "Season and limited time",
                 "Location", "Discovery stack", "Unclassifiable"]
        palette = ["#2E5248", "#436B5E", "#5E8172", "#7E9A8B", "#9DB3A6",
                   "#8C6A3F", "#B09267", "#C9B693", "#CBC6B8", "#E6E2D8"]

        mix = (pd.crosstab(coded["hotel"], coded["category"], normalize="index")
               * 100).reindex(HOTEL_ORDER)

        st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
        st.markdown("## What each house tags about")
        c1, c2 = st.columns([3, 2], gap="large")
        with c1:
            ml = mix.reset_index().melt(id_vars="hotel", var_name="category",
                                        value_name="pct")
            st.altair_chart(
                alt.Chart(ml).mark_bar(height=22).encode(
                    x=alt.X("pct:Q", stack="normalize", title=None,
                            axis=axis_num(format="%")),
                    y=alt.Y("hotel:N", title=None, sort=HOTEL_ORDER, axis=axis_cat()),
                    color=alt.Color("category:N",
                                    scale=alt.Scale(domain=order, range=palette),
                                    legend=alt.Legend(title=None, orient="bottom",
                                                      columns=4, labelColor=MUTED)),
                    order=alt.Order("category:N"),
                    tooltip=["hotel", "category", alt.Tooltip("pct:Q", format=".1f")],
                ).properties(height=250).configure_view(strokeWidth=0),
                use_container_width=True)
        with c2:
            bl = (mix["Brand philosophy"].idxmax()
                  if "Brand philosophy" in mix.columns else None)
            gl = (mix["Discovery stack"].idxmax()
                  if "Discovery stack" in mix.columns else None)
            st.markdown(
                f"<div class='read'>"
                f"<p>Every distinct hashtag was coded against a written codebook. "
                f"The mix is each house's total hashtag usage by category.</p>"
                f"<p>{SHORT.get(bl, '—')} commits the largest share to proprietary "
                f"brand language — coined phrases no competitor can use. "
                f"{SHORT.get(gl, '—')} commits the most to the shared discovery stack: "
                f"reach that belongs to no one in particular.</p></div>",
                unsafe_allow_html=True)
            paper_note("<b>Paper:</b> the typology section. Brand, place and "
                       "experiential tags carry distinct symbolic functions; this "
                       "is the evidence that rival houses in one destination "
                       "distribute them differently.")

        # -- Construal ------------------------------------------------------
        cons = coded[coded["category"].isin(ABSTRACT | CONCRETE)].copy()
        if len(cons):
            cons["level"] = np.where(cons["category"].isin(ABSTRACT),
                                     "Abstract", "Concrete")
            cm = (pd.crosstab(cons["hotel"], cons["level"], normalize="index")
                  * 100).reindex(HOTEL_ORDER).reset_index()
            if "Abstract" in cm.columns:
                st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
                st.markdown("## Distance, measured")
                c1, c2 = st.columns([3, 2], gap="large")
                with c1:
                    st.altair_chart(
                        ranked_bar(cm.rename(columns={"Abstract": "abstract"}),
                                   "abstract",
                                   "Share of tags that are abstract (%)", fmt=".1f"),
                        use_container_width=True)
                with c2:
                    top = cm.loc[cm["Abstract"].idxmax()]
                    bot = cm.loc[cm["Abstract"].idxmin()]
                    st.markdown(
                        f"<div class='read'>"
                        f"<p>Brand philosophy and cultural-register tags describe "
                        f"what a stay <em>means</em>. Outlets, occasions, places and "
                        f"seasons name something a guest could point at or book. The "
                        f"first construes the property at a distance; the second "
                        f"brings it close. Identity and discovery tags do neither and "
                        f"are excluded here.</p>"
                        f"<p>{SHORT[top['hotel']]} tags most abstractly "
                        f"({top['Abstract']:.0f}%); {SHORT[bot['hotel']]} most "
                        f"concretely ({bot['Abstract']:.0f}% abstract). If "
                        f"psychological distance sustains luxury value, this is "
                        f"where it becomes visible in daily practice.</p></div>",
                        unsafe_allow_html=True)
                    paper_note("<b>Paper:</b> the construal-level hypothesis made "
                               "operational. Abstract tagging is the linguistic "
                               "trace of the symbolic distance the review argues "
                               "luxury brands must protect while remaining "
                               "present on an open platform.")

    # -- Japan: script over time --------------------------------------------
    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
    st.markdown("## Which audience is being tagged for")

    qt = long.copy()
    qt["quarter"] = qt["date"].dt.to_period("Q").dt.to_timestamp()
    q = (qt.assign(is_jp=qt["script"].eq("Japanese"))
         .groupby("quarter").agg(jp_pct=("is_jp", lambda s: s.mean() * 100),
                                 n=("is_jp", "size")).reset_index())
    q = q[q["n"] >= 100].copy()
    # Vega's time format has no quarter token, so build the label ourselves.
    q["label"] = (q["quarter"].dt.year.astype(str) + " Q"
                  + q["quarter"].dt.quarter.astype(str))

    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        if len(q) >= 3:
            st.altair_chart(
                alt.Chart(q).mark_line(
                    color=ACCENT, strokeWidth=2,
                    point=alt.OverlayMarkDef(color=ACCENT, size=34)
                ).encode(
                    x=alt.X("label:N", title=None, sort=list(q["label"]),
                            axis=axis_cat()),
                    y=alt.Y("jp_pct:Q", title="Japanese-script tags (%)",
                            scale=alt.Scale(zero=False), axis=axis_num()),
                    tooltip=[alt.Tooltip("label:N", title="Quarter"),
                             alt.Tooltip("jp_pct:Q", format=".1f"),
                             alt.Tooltip("n:Q", title="tags")],
                ).properties(height=260).configure_view(strokeWidth=0),
                use_container_width=True)
        else:
            st.info("Not enough quarters in this window to plot a trend.")
    with c2:
        if len(q) >= 3:
            peak, last = q.loc[q["jp_pct"].idxmax()], q.iloc[-1]
            st.markdown(
                f"<div class='read'>"
                f"<p>Japanese-script tagging peaked at {peak['jp_pct']:.0f}% in "
                f"{peak['label']} and stands at {last['jp_pct']:.0f}% by "
                f"{last['label']} — a swing of "
                f"{peak['jp_pct'] - last['jp_pct']:.0f} points toward Latin script "
                f"across the whole set.</p>"
                f"<p>Script is not decoration in this market. Japanese users search "
                f"Instagram by hashtag at several times the global rate, so the "
                f"script chosen decides which of two audiences can find the post "
                f"at all.</p></div>", unsafe_allow_html=True)
        paper_note("<b>Paper:</b> the destination-specific contribution. Curated "
                   "visibility in a bilingual inbound market has an axis the "
                   "existing literature does not model — whom the curation is "
                   "legible to. The trend runs alongside the inbound recovery "
                   "described in the introduction.")

    # -- Japan: bilingual norm ----------------------------------------------
    ps = post_script.copy()
    ps["mode"] = np.where(ps["jp_share"] == 0, "Latin only",
                          np.where(ps["jp_share"] == 1, "Japanese only", "Mixed"))
    mode_mix = (pd.crosstab(ps["hotel"], ps["mode"], normalize="index")
                * 100).reindex(HOTEL_ORDER)

    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        mm = mode_mix.reset_index().melt(id_vars="hotel", var_name="mode",
                                         value_name="pct")
        st.altair_chart(
            alt.Chart(mm).mark_bar(height=20).encode(
                x=alt.X("pct:Q", stack="normalize", title=None,
                        axis=axis_num(format="%")),
                y=alt.Y("hotel:N", title=None, sort=HOTEL_ORDER, axis=axis_cat()),
                color=alt.Color("mode:N",
                                scale=alt.Scale(domain=["Japanese only", "Mixed",
                                                        "Latin only"],
                                                range=[ACCENT, "#A9BCB0", WARM]),
                                legend=alt.Legend(title=None, orient="bottom",
                                                  labelColor=MUTED)),
                tooltip=["hotel", "mode", alt.Tooltip("pct:Q", format=".1f")],
            ).properties(height=230).configure_view(strokeWidth=0),
            use_container_width=True)
    with c2:
        mixed = mode_mix["Mixed"].mean() if "Mixed" in mode_mix.columns else 0
        st.markdown(
            f"<div class='read'>"
            f"<p>Inside a single post the default is bilingual: on average "
            f"{mixed:.0f}% of tagged posts carry both scripts at once. Hedging is "
            f"the house style.</p>"
            f"<p>Two properties have opted out, in opposite directions. Janu tags "
            f"almost entirely in Latin script; the Peninsula, across its small "
            f"number of tagged posts, leans the other way.</p></div>",
            unsafe_allow_html=True)
        paper_note("<b>Paper:</b> post-level evidence that audience targeting is "
                   "resolved inside the post rather than across the account — "
                   "support for treating the post as the unit of analysis.")

    # -- Engagement ---------------------------------------------------------
    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
    st.markdown("## More tags does not mean more engagement")

    corr = pd.DataFrame([{
        "hotel": h,
        "Likes": rank_corr(posts[posts["hotel"] == h]["tag_count"],
                           posts[posts["hotel"] == h]["likes"]),
        "Comments": rank_corr(posts[posts["hotel"] == h]["tag_count"],
                              posts[posts["hotel"] == h]["comments"]),
    } for h in HOTEL_ORDER])

    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        cm2 = corr.melt(id_vars="hotel", var_name="metric",
                        value_name="rho").dropna()
        if len(cm2):
            st.altair_chart(
                alt.Chart(cm2).mark_bar(height=12).encode(
                    x=alt.X("rho:Q", title="Rank correlation with hashtag count",
                            scale=alt.Scale(domain=[-.35, .35]), axis=axis_num()),
                    y=alt.Y("hotel:N", title=None, sort=HOTEL_ORDER, axis=axis_cat()),
                    yOffset=alt.YOffset("metric:N"),
                    color=alt.Color("metric:N",
                                    scale=alt.Scale(domain=["Likes", "Comments"],
                                                    range=[ACCENT, WARM]),
                                    legend=alt.Legend(title=None, orient="top",
                                                      labelColor=MUTED)),
                    tooltip=["hotel", "metric", alt.Tooltip("rho:Q", format=".3f")],
                ).properties(height=250).configure_view(strokeWidth=0),
                use_container_width=True)
        else:
            st.info("Too few posts in this window to estimate correlations.")
    with c2:
        st.markdown(
            "<div class='read'>"
            "<p>Each bar is measured inside a single account. Comparing across "
            "hotels would only measure who has more followers, and follower "
            "counts are not observable from public posts.</p>"
            "<p>The effect splits by house and by metric. On several accounts more "
            "tags accompany modestly more likes. On the most established accounts "
            "likes are flat while comments fall — reach and conversation are not "
            "the same outcome, and tagging appears to trade one against the "
            "other.</p></div>", unsafe_allow_html=True)
        paper_note("<b>Paper:</b> the engagement question answered without "
                   "overclaiming. A null-to-negative result on the most "
                   "prestigious accounts is what distinguishes luxury from "
                   "mainstream digital marketing, where reach is treated as an "
                   "unqualified good.")

    st.markdown("<p class='note'>Correlational, observational data. Subject, "
                "timing and format all move engagement, and none are controlled "
                "here.</p>", unsafe_allow_html=True)

    # -- Format -------------------------------------------------------------
    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
    st.markdown("## Format, for context")
    fmt = (posts.groupby("format")
           .agg(posts=("likes", "size"), median_likes=("likes", "median"))
           .reset_index().sort_values("median_likes", ascending=False))
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.altair_chart(
            alt.Chart(fmt).mark_bar(height=24).encode(
                x=alt.X("median_likes:Q", title="Median likes", axis=axis_num()),
                y=alt.Y("format:N", title=None, sort="-x", axis=axis_cat()),
                color=alt.value(ACCENT),
                tooltip=["format", "posts", "median_likes"],
            ).properties(height=150).configure_view(strokeWidth=0),
            use_container_width=True)
    with c2:
        ranking = " → ".join(f"{r['format']} ({r['median_likes']:.0f})"
                             for _, r in fmt.iterrows())
        video_rank = (list(fmt["format"]).index("Video") + 1
                      if "Video" in list(fmt["format"]) else None)
        video_line = (
            f"Video ranks {video_rank} of {len(fmt)} on median likes here."
            if video_rank else "")
        st.markdown(
            f"<div class='read'>"
            f"<p>By median likes: {ranking}. {video_line}</p>"
            f"<p>One caveat, stated plainly: this sample is feed posts only. "
            f"Reels are excluded, so this compares video inside the feed against "
            f"stills, not the whole video strategy.</p></div>",
            unsafe_allow_html=True)

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
            return st.secrets["OPENAI_API_KEY"]
    except Exception:            # no secrets.toml locally
        pass
    import os
    return os.environ.get("OPENAI_API_KEY")


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
    if not key:
        st.warning("No OPENAI_API_KEY found. Add it under Manage app → Settings → "
                   "Secrets as:  OPENAI_API_KEY = \"sk-...\"")
    else:
        st.markdown(
            f"<p class='note'>Classifies all {len(tag_counts):,} tags against the "
            f"codebook in classify_hashtags.py, in batches of {ch.BATCH_SIZE} at "
            f"temperature 0. A few minutes and a few cents on "
            f"{ch.MODEL}.</p>", unsafe_allow_html=True)

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
                bar = st.progress(0.0, text="Starting…")

                for start in range(0, len(todo), ch.BATCH_SIZE):
                    batch = todo[start:start + ch.BATCH_SIZE]
                    try:
                        result = ch.classify_batch(client, batch)
                    except Exception:
                        failed += len(batch)
                        result = {}
                    for tag in batch:
                        entry = result.get(tag) or {}
                        cat = entry.get("category", "Unclassifiable")
                        if cat not in ch.CATEGORIES:
                            cat = "Unclassifiable"
                        done[tag] = (cat, entry.get("confidence", "low"))
                    seen = min(start + ch.BATCH_SIZE, len(todo))
                    bar.progress(seen / len(todo), text=f"{seen:,} of {len(todo):,}")

                bar.empty()
                out = tag_counts.copy()
                out["category"] = out["tag"].map(lambda t: done.get(t, ("Unclassifiable",))[0])
                out["confidence"] = out["tag"].map(lambda t: done.get(t, (None, "low"))[1])
                st.session_state["classified"] = out
                if failed:
                    st.warning(f"{failed} tags fell back to Unclassifiable after a "
                               f"failed request. Re-run to retry them.")

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
        f"<p class='note'><b>Windows.</b> The full window begins May 2023. The "
        f"matched window begins 18 February 2026 and is reported as a robustness "
        f"check; findings that hold in both are identified as such.</p>"
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
