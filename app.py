import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# -----------------------------
# Page config + minimal styling
# -----------------------------
st.set_page_config(
    page_title="Luxury Hashtag Intelligence — Tokyo",
    page_icon="🏨",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.15rem; padding-bottom: 2.2rem; max-width: 1280px; }
      h1, h2, h3 { letter-spacing: 0.2px; }
      .small-note { opacity: 0.82; font-size: 0.92rem; }
      div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        padding: 12px 14px;
        border-radius: 14px;
      }
      .card {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        padding: 14px 16px;
        border-radius: 14px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Accounts included (strict scope)
# -----------------------------
ALLOWED_ACCOUNTS = [
    "thepeninsulatokyo",
    "aman_tokyo",
    "ritzcarltontokyo",
    "fstokyo",
    "thecapitolhoteltokyu",
    "parkhyatttokyo",
    "janutokyo",
]

# -----------------------------
# Helpers
# -----------------------------
def _sorted_hashtag_cols(cols):
    # Keep hashtags/0 ... hashtags/29 in numeric order if present
    hc = [c for c in cols if c.startswith("hashtags/")]
    def key(c):
        try:
            return int(c.split("/")[-1])
        except:
            return 9999
    return sorted(hc, key=key)

def classify_hashtag_theme(tag: str) -> str:
    """
    Transparent, editable taxonomy (descriptive, not evaluative).
    This is NOT sentiment/demographics; it's a language bucket.
    """
    if not isinstance(tag, str) or tag == "":
        return "None"
    t = tag.lower()

    # Brand / Property anchors (you can refine as you learn)
    if any(k in t for k in ["aman", "parkhyatt", "capitol", "janu", "peninsula", "fourseasons", "ritz", "fs", "fstokyo"]):
        return "Brand / Property"

    # Place / Location
    if any(k in t for k in ["tokyo", "japan", "nihonbashi", "marunouchi"]):
        return "Place / Location"

    # Dining / Wellness
    if any(k in t for k in ["dining", "restaurant", "bar", "lounge", "cafe", "wellness", "spa"]):
        return "Dining / Wellness"

    # Experience / Atmosphere
    if any(k in t for k in ["design", "architecture", "interior", "experience", "moments", "stay"]):
        return "Experience / Atmosphere"

    # Generic Travel / Lifestyle (broadcast-leaning language)
    if any(k in t for k in ["travel", "wanderlust", "vacation", "luxury", "hotel", "trip", "tourism", "visit"]):
        return "Generic Travel / Lifestyle"

    return "Other"

def herfindahl_index(values: pd.Series) -> float:
    if len(values) == 0:
        return 0.0
    counts = values.value_counts(normalize=True)
    return float((counts ** 2).sum())

def safe_date_range(date_value, min_dt, max_dt):
    """
    Streamlit date_input can return a single date or a tuple/list.
    This makes it robust and prevents scope-edit errors.
    """
    if isinstance(date_value, (list, tuple)) and len(date_value) == 2:
        start, end = date_value[0], date_value[1]
    else:
        # If user picks a single date, treat it as a 1-day window
        start = date_value
        end = date_value
    # Clamp just in case
    start = max(start, min_dt.date())
    end = min(end, max_dt.date())
    return start, end

# -----------------------------
# Data loading
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    # Robust encoding (Instagram captions often break strict UTF-8)
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")

    # Drop common CSV artifact columns if present
    df = df.loc[:, ~df.columns.str.startswith("Unnamed:")].copy()

    required = {"ownerUsername", "timestamp", "caption", "likesCount", "commentsCount"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(list(missing))}")

    hashtag_cols = _sorted_hashtag_cols(df.columns)
    if not hashtag_cols:
        raise ValueError("No hashtag columns found (expected hashtags/0 … hashtags/29).")

    df = df.copy()

    # Normalize usernames
    df["ownerUsername"] = df["ownerUsername"].astype(str).str.strip().str.lower()

    # Strict scope
    df = df[df["ownerUsername"].isin([a.lower() for a in ALLOWED_ACCOUNTS])].copy()

    # Timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    if df["timestamp"].isna().all():
        raise ValueError("Could not parse 'timestamp'. Check the timestamp format in your CSV.")

    # Numeric
    df["likesCount"] = pd.to_numeric(df["likesCount"], errors="coerce").fillna(0).astype(int)
    df["commentsCount"] = pd.to_numeric(df["commentsCount"], errors="coerce").fillna(0).astype(int)

    # Hashtag normalization (no invented meaning, just normalization)
    for c in hashtag_cols:
        df[c] = df[c].astype(str).str.strip()
        df.loc[df[c].isin(["", "nan", "None", "NaN"]), c] = ""
        df[c] = df[c].str.lower().str.replace("#", "", regex=False)

    # Allowed derived metrics
    df["total_engagement"] = df["likesCount"] + df["commentsCount"]
    df["comment_like_ratio"] = df["commentsCount"] / df["likesCount"].replace(0, np.nan)
    df["comment_like_ratio"] = df["comment_like_ratio"].fillna(0.0)

    df["hashtag_count"] = df[hashtag_cols].apply(
        lambda row: sum(1 for v in row.values if isinstance(v, str) and v != ""),
        axis=1
    )

    return df

def explode_hashtags(df: pd.DataFrame) -> pd.DataFrame:
    hashtag_cols = _sorted_hashtag_cols(df.columns)
    long = df[
        ["ownerUsername", "timestamp", "likesCount", "commentsCount", "total_engagement", "comment_like_ratio", "hashtag_count"] + hashtag_cols
    ].melt(
        id_vars=["ownerUsername", "timestamp", "likesCount", "commentsCount", "total_engagement", "comment_like_ratio", "hashtag_count"],
        value_vars=hashtag_cols,
        var_name="hashtag_slot",
        value_name="hashtag",
    )
    long = long[long["hashtag"].astype(str).str.len() > 0].copy()
    long["hashtag"] = long["hashtag"].astype(str)
    long["hashtag_theme"] = long["hashtag"].apply(classify_hashtag_theme)
    return long

def row_tag_set(row_vals):
    return set(
        [str(x).strip().lower().replace("#", "") for x in row_vals if isinstance(x, str) and str(x).strip() != ""]
    )

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Controls")
st.sidebar.markdown(
    '<div class="small-note">Strategy-first controls: keep scope comparable across properties.</div>',
    unsafe_allow_html=True
)

data_path = st.sidebar.text_input("CSV path", value="instagramscraperfile.csv")
try:
    df = load_data(data_path)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

min_dt = df["timestamp"].min().to_pydatetime()
max_dt = df["timestamp"].max().to_pydatetime()

date_value = st.sidebar.date_input(
    "Date range (UTC)",
    value=(min_dt.date(), max_dt.date()),
    min_value=min_dt.date(),
    max_value=max_dt.date(),
)
start_d, end_d = safe_date_range(date_value, min_dt, max_dt)
start_date = pd.to_datetime(start_d).tz_localize("UTC")
end_date = pd.to_datetime(end_d).tz_localize("UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

hotels = st.sidebar.multiselect(
    "Hotels (accounts)",
    options=[a.lower() for a in ALLOWED_ACCOUNTS],
    default=[a.lower() for a in ALLOWED_ACCOUNTS],
)

min_posts = st.sidebar.slider(
    "Minimum posts per hotel (for comparisons)",
    min_value=10, max_value=300, value=30, step=10
)

# Dilution audit list (editable; avoids assumptions)
default_dilution = [
    "travel", "instatravel", "travelgram", "wanderlust", "vacation",
    "trip", "tourism", "hotel", "luxury", "luxuryhotel",
    "tokyo", "japan", "visitjapan", "beautifuldestinations"
]
dilution_text = st.sidebar.text_area(
    "Dilution-audit tags (editable)\n(one per line, no #)",
    value="\n".join(default_dilution),
    height=170,
)
dilution_tags = [t.strip().lower().replace("#", "") for t in dilution_text.splitlines() if t.strip()]

# Apply filters
f = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)].copy()
f = f[f["ownerUsername"].isin(hotels)].copy()

if len(f) == 0:
    st.warning("No rows after filters. Expand date range or include more hotels.")
    st.stop()

# Long hashtag table
hl = explode_hashtags(f)

# Post-level tag sets (for fast checks)
hashtag_cols = _sorted_hashtag_cols(f.columns)
tag_sets = f[hashtag_cols].apply(lambda r: row_tag_set(r.values), axis=1)

# Broadcast signal: share of “audit” tags among tags used in a post (0 if no hashtags)
def broadcast_share(tagset):
    if not tagset or len(tagset) == 0:
        return 0.0
    return sum(1 for t in tagset if t in set(dilution_tags)) / len(tagset)

f = f.copy()
f["broadcast_share"] = tag_sets.apply(broadcast_share)
f["has_any_audit_tag"] = tag_sets.apply(lambda s: any(t in s for t in dilution_tags))

# Assign a simple "language context" per post based on which theme appears most among its hashtags
# (Descriptive only; helps within-theme comparisons without using content/sentiment.)
theme_counts = (
    hl.groupby(["ownerUsername", "timestamp"])
      .pivot_table(index=["ownerUsername", "timestamp"], columns="hashtag_theme", values="hashtag", aggfunc="count", fill_value=0)
      .reset_index()
)
f = f.merge(theme_counts, on=["ownerUsername", "timestamp"], how="left")
theme_cols = [c for c in theme_counts.columns if c not in ["ownerUsername", "timestamp"]]

def dominant_theme(row):
    if not theme_cols:
        return "None"
    counts = {c: row.get(c, 0) for c in theme_cols}
    # If no hashtags at all
    if sum(counts.values()) == 0:
        return "None"
    return max(counts, key=counts.get)

f["dominant_hashtag_theme"] = f.apply(dominant_theme, axis=1)

# -----------------------------
# Header
# -----------------------------
st.title("Luxury Hashtag Intelligence — Tokyo")
st.markdown(
    """
    **Purpose:** treat hashtags as **brand language** (signals of restraint and identity), not growth tactics.  
    **Engagement depth:** we use **comment behavior** (counts, not text) and **consistency** as a proxy for guest consideration.
    """
)

# KPI strip
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Posts (filtered)", f"{len(f):,}")
k2.metric("Hotels included", f"{f['ownerUsername'].nunique():,}")
k3.metric("Median hashtags / post", f"{int(f['hashtag_count'].median())}")
k4.metric("Median comments", f"{int(f['commentsCount'].median())}")
k5.metric("Median comment/like ratio", f"{f['comment_like_ratio'].median():.3f}")

st.divider()

# Tabs (aligned to your research questions, with safer wording)
tab1, tab2, tab3, tab4 = st.tabs([
    "1) High-Intent Language (Themes)",
    "2) Restraint vs Broadcast Signals",
    "3) Dilution Risk Audit (Associations)",
    "4) Hotel Hashtag Fingerprints",
])

# ---------------------------------
# TAB 1: High-intent language (themes)
# ---------------------------------
with tab1:
    st.subheader("Which types of hashtag language are associated with higher-intent engagement?")
    st.markdown(
        "To avoid over-claiming causality, we evaluate **themes of hashtag language** (descriptive categories) "
        "and how they are **associated** with engagement depth and consistency."
    )

    overall_med = f["comment_like_ratio"].median()

    # Theme-level aggregation
    theme_agg = hl.groupby("hashtag_theme").agg(
        hashtag_instances=("hashtag_theme", "size"),
        hotels=("ownerUsername", "nunique"),
        median_ratio=("comment_like_ratio", "median"),
        mean_ratio=("comment_like_ratio", "mean"),
    ).reset_index()

    # Consistency: share of instances above overall median depth
    tmp = hl.copy()
    tmp["above_median_depth"] = (tmp["comment_like_ratio"] >= overall_med).astype(int)
    cons = tmp.groupby("hashtag_theme")["above_median_depth"].mean().reset_index().rename(columns={"above_median_depth": "consistency_score"})
    theme_agg = theme_agg.merge(cons, on="hashtag_theme", how="left")

    # Display table
    st.markdown("**Theme summary (decision-ready):**")
    st.dataframe(
        theme_agg.sort_values("median_ratio", ascending=False)
                 .round({"median_ratio": 4, "mean_ratio": 4, "consistency_score": 3}),
        use_container_width=True,
        height=320
    )

    # Scatter with quadrant lines (avg depth + avg consistency)
    avg_depth = float(theme_agg["median_ratio"].mean()) if len(theme_agg) else 0.0
    avg_cons = float(theme_agg["consistency_score"].mean()) if len(theme_agg) else 0.0

    chart_base = alt.Chart(theme_agg).mark_circle(size=260).encode(
        x=alt.X("consistency_score:Q", title="Consistency (share of instances above overall median depth)"),
        y=alt.Y("median_ratio:Q", title="Median comment/like ratio (depth)"),
        size=alt.Size("hashtag_instances:Q", title="Hashtag instances"),
        tooltip=[
            "hashtag_theme",
            "hashtag_instances",
            "hotels",
            alt.Tooltip("median_ratio:Q", format=".4f"),
            alt.Tooltip("consistency_score:Q", format=".3f"),
        ],
    )

    vline = alt.Chart(pd.DataFrame({"x": [avg_cons]})).mark_rule(strokeDash=[4,4], color="gray").encode(x="x:Q")
    hline = alt.Chart(pd.DataFrame({"y": [avg_depth]})).mark_rule(strokeDash=[4,4], color="gray").encode(y="y:Q")

    st.markdown("**Theme map (quadrants highlight what’s above-average in depth and consistency):**")
    st.altair_chart((chart_base + vline + hline).interactive().properties(height=460), use_container_width=True)

    st.info(
        "Interpretation guidance: themes that sit **above-average in both depth and consistency** suggest a more intentional engagement context. "
        "This is an association, not a claim that language alone drives comments."
    )

# ---------------------------------
# TAB 2: Restraint vs Broadcast signals
# ---------------------------------
with tab2:
    st.subheader("How do restraint and broadcast signals show up in hashtag behavior?")
    st.markdown(
        "Hashtag volume alone does not define broadcast vs restraint. Here we separate **how much hotels tag** "
        "from **what kinds of tags they use**, and treat **zero-hashtag posts** as a distinct signaling context."
    )

    # Hashtag count distribution (bar; bins are categorical)
    bins = [-0.1, 0.1, 3, 5, 8, 12, 20, 30]
    labels = ["0", "1–3", "4–5", "6–8", "9–12", "13–20", "21–30"]
    f2 = f.copy()
    f2["hashtag_bin"] = pd.cut(f2["hashtag_count"], bins=bins, labels=labels, include_lowest=True)

    bin_counts = f2.groupby("hashtag_bin").size().reset_index(name="posts")

    left, right = st.columns([1.05, 0.95])

    with left:
        st.markdown("**How often each hashtag volume band occurs**")
        st.altair_chart(
            alt.Chart(bin_counts.dropna()).mark_bar().encode(
                x=alt.X("hashtag_bin:N", title="Hashtag volume (binned)", sort=labels),
                y=alt.Y("posts:Q", title="Number of posts"),
                tooltip=["hashtag_bin", "posts"],
            ).properties(height=320),
            use_container_width=True
        )
        st.caption("Bins are categorical; bar charts avoid implying continuity.")

    with right:
        st.markdown("**Engagement depth by volume band (descriptive)**")
        overall = f2.groupby("hashtag_bin").agg(
            posts=("hashtag_bin", "size"),
            median_ratio=("comment_like_ratio", "median"),
            median_comments=("commentsCount", "median"),
            median_broadcast_share=("broadcast_share", "median"),
        ).reset_index()

        st.dataframe(
            overall.dropna().round({"median_ratio": 4, "median_broadcast_share": 3}),
            use_container_width=True,
            height=320
        )
        st.caption("Includes broadcast-share context to avoid equating volume with broadcast behavior.")

    st.divider()

    st.markdown("**By hotel: restraint + broadcast profile**")
    hotel_profile = f2.groupby("ownerUsername").agg(
        posts=("ownerUsername", "size"),
        median_hashtags=("hashtag_count", "median"),
        pct_zero_hashtags=("hashtag_count", lambda s: float((s == 0).mean())),
        median_ratio=("comment_like_ratio", "median"),
        median_broadcast_share=("broadcast_share", "median"),
    ).reset_index()

    hotel_profile = hotel_profile[hotel_profile["posts"] >= min_posts].copy()

    st.dataframe(
        hotel_profile.sort_values(["median_hashtags", "median_broadcast_share"])
                     .round({"pct_zero_hashtags": 3, "median_ratio": 4, "median_broadcast_share": 3}),
        use_container_width=True,
        height=360
    )

    st.info(
        "From a brand-signaling perspective: a hotel can use more hashtags and still remain restrained if its language stays specific and anchored. "
        "Conversely, even a small number of generic tags can read as broadcast."
    )

# ---------------------------------
# TAB 3: Dilution risk audit (associations; within-theme)
# ---------------------------------
with tab3:
    st.subheader("Which hashtags are associated with diluted engagement depth?")
    st.markdown(
        "This is an **audit**, not a judgment: you define potentially mass-market tags, and we test whether they are **associated** "
        "with shallower engagement depth in this dataset."
    )

    if len(dilution_tags) == 0:
        st.warning("Add at least 1 dilution-audit tag in the sidebar to run this section.")
        st.stop()

    # Overall: with vs without audit tags
    grp = f.groupby("has_any_audit_tag").agg(
        posts=("has_any_audit_tag", "size"),
        median_ratio=("comment_like_ratio", "median"),
        median_comments=("commentsCount", "median"),
        median_hashtags=("hashtag_count", "median"),
        median_broadcast_share=("broadcast_share", "median"),
    ).reset_index()

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Overall association: posts with vs without audit tags**")
        st.dataframe(
            grp.rename(columns={"has_any_audit_tag": "contains_audit_tags"})
               .round({"median_ratio": 4, "median_broadcast_share": 3}),
            use_container_width=True,
            height=240
        )
        st.caption("We use 'associated with' to avoid implying causality.")

    with c2:
        st.markdown("**Audit-tag presence by hotel**")
        by_hotel = f.groupby("ownerUsername").agg(
            posts=("ownerUsername", "size"),
            pct_with_audit_tags=("has_any_audit_tag", lambda s: float(s.mean())),
            median_ratio=("comment_like_ratio", "median"),
        ).reset_index()
        st.dataframe(by_hotel.round({"pct_with_audit_tags": 3, "median_ratio": 4}), use_container_width=True, height=240)

    st.divider()

    # Within-theme comparison (dominant hashtag theme)
    st.markdown("**Within-language-context comparison (reduces content-mix bias)**")
    available_themes = sorted(f["dominant_hashtag_theme"].dropna().unique().tolist())
    picked_theme = st.selectbox("Choose a dominant hashtag theme context", options=available_themes, index=0)

    within = f[f["dominant_hashtag_theme"] == picked_theme].copy()
    if len(within) < 30:
        st.warning("This theme context has a small sample size after filters. Interpret cautiously or widen the date range.")
    within_grp = within.groupby("has_any_audit_tag").agg(
        posts=("has_any_audit_tag", "size"),
        median_ratio=("comment_like_ratio", "median"),
        median_comments=("commentsCount", "median"),
        median_hashtags=("hashtag_count", "median"),
    ).reset_index()

    st.dataframe(
        within_grp.rename(columns={"has_any_audit_tag": "contains_audit_tags"}).round({"median_ratio": 4}),
        use_container_width=True
    )
    st.caption("This controls for broad language context using hashtag-derived themes (descriptive, not content sentiment).")

    # Tag-level audit detail
    hl2 = hl.copy()
    hl2["is_audit_tag"] = hl2["hashtag"].isin(set(dilution_tags))

    audit_detail = hl2[hl2["is_audit_tag"]].groupby("hashtag").agg(
        uses=("hashtag", "size"),
        hotels=("ownerUsername", "nunique"),
        median_ratio=("comment_like_ratio", "median"),
        mean_ratio=("comment_like_ratio", "mean"),
        avg_comments=("commentsCount", "mean"),
    ).reset_index().sort_values(["median_ratio", "uses"], ascending=[True, False])

    st.markdown("**Audit-tag detail (lowest depth rises to top)**")
    st.dataframe(
        audit_detail.round({"median_ratio": 4, "mean_ratio": 4, "avg_comments": 2}),
        use_container_width=True,
        height=420
    )

    st.info(
        "Interpretation: if a generic tag is consistently associated with shallower engagement depth, "
        "it may act as a broadcast signal. This suggests using it deliberately (campaign contexts) rather than as default brand vocabulary."
    )

# ---------------------------------
# TAB 4: Hotel hashtag fingerprints (signal language)
# ---------------------------------
with tab4:
    st.subheader("How do different hotels signal brand language through hashtags?")
    st.markdown(
        "We summarize each hotel’s hashtag “fingerprint” using how much they tag, what kinds of tags they rely on, "
        "and which repeated anchor tags define their vocabulary — descriptively, not as a value judgment."
    )

    hotel_hashtags = hl.groupby("ownerUsername")["hashtag"].apply(list).reset_index()

    rows = []
    for _, r in hotel_hashtags.iterrows():
        hotel = r["ownerUsername"]
        tags = pd.Series(r["hashtag"])
        hhi = herfindahl_index(tags)
        unique = int(tags.nunique())
        top_tags = tags.value_counts().head(8).index.tolist()

        sub = f[f["ownerUsername"] == hotel].copy()

        # Theme mix (share of hashtag instances by theme)
        sub_hl = hl[hl["ownerUsername"] == hotel].copy()
        theme_mix = (sub_hl["hashtag_theme"].value_counts(normalize=True).to_dict() if len(sub_hl) else {})

        rows.append({
            "ownerUsername": hotel,
            "posts": int(len(sub)),
            "median_hashtag_count": float(sub["hashtag_count"].median()) if len(sub) else 0.0,
            "pct_zero_hashtags": float((sub["hashtag_count"] == 0).mean()) if len(sub) else 0.0,
            "median_comment_like_ratio": float(sub["comment_like_ratio"].median()) if len(sub) else 0.0,
            "median_broadcast_share": float(sub["broadcast_share"].median()) if len(sub) else 0.0,
            "unique_hashtags_used": unique,
            "language_concentration_hhi": hhi,
            "top_hashtag_anchors": ", ".join(top_tags),
            "theme_mix_brand_property": float(theme_mix.get("Brand / Property", 0.0)),
            "theme_mix_place": float(theme_mix.get("Place / Location", 0.0)),
            "theme_mix_dining_wellness": float(theme_mix.get("Dining / Wellness", 0.0)),
            "theme_mix_experience": float(theme_mix.get("Experience / Atmosphere", 0.0)),
            "theme_mix_generic_travel": float(theme_mix.get("Generic Travel / Lifestyle", 0.0)),
        })

    fp = pd.DataFrame(rows)
    fp = fp[fp["posts"] >= min_posts].copy()

    c1, c2 = st.columns([1.15, 0.85])

    with c1:
        st.markdown("**Fingerprint table (director-friendly)**")
        st.dataframe(
            fp.sort_values(["median_hashtag_count", "median_broadcast_share"], ascending=[True, True])
              .round({
                  "pct_zero_hashtags": 3,
                  "median_comment_like_ratio": 4,
                  "median_broadcast_share": 3,
                  "language_concentration_hhi": 4,
                  "theme_mix_brand_property": 3,
                  "theme_mix_place": 3,
                  "theme_mix_dining_wellness": 3,
                  "theme_mix_experience": 3,
                  "theme_mix_generic_travel": 3,
              }),
            use_container_width=True,
            height=520
        )
        st.caption(
            "HHI: higher values mean a tighter, repeated hashtag vocabulary. Theme mix shows what kinds of language dominate the hotel’s tags."
        )

    with c2:
        st.markdown("**Restraint vs Depth (hotel-level)**")
        if len(fp) >= 2:
            avg_depth = float(fp["median_comment_like_ratio"].mean())
            avg_tags = float(fp["median_hashtag_count"].mean())

            base = alt.Chart(fp).mark_circle(size=190).encode(
                x=alt.X("median_hashtag_count:Q", title="Median hashtags / post"),
                y=alt.Y("median_comment_like_ratio:Q", title="Median comment/like ratio"),
                tooltip=[
                    "ownerUsername", "posts",
                    alt.Tooltip("median_hashtag_count:Q", format=".1f"),
                    alt.Tooltip("median_comment_like_ratio:Q", format=".4f"),
                    alt.Tooltip("median_broadcast_share:Q", format=".3f"),
                ],
            )

            vline = alt.Chart(pd.DataFrame({"x": [avg_tags]})).mark_rule(strokeDash=[4,4], color="gray").encode(x="x:Q")
            hline = alt.Chart(pd.DataFrame({"y": [avg_depth]})).mark_rule(strokeDash=[4,4], color="gray").encode(y="y:Q")

            st.altair_chart((base + vline + hline).interactive().properties(height=520), use_container_width=True)
        else:
            st.info("Not enough hotels meet the minimum-post threshold for this comparison. Lower the slider in the sidebar.")

    st.divider()

    st.markdown("**Hotel detail (for citations in a research report)**")
    chosen = st.selectbox("Select one hotel", options=sorted(f["ownerUsername"].unique().tolist()))
    sub = f[f["ownerUsername"] == chosen].copy()
    sub_hl = hl[hl["ownerUsername"] == chosen].copy()

    # Anchors by depth association (still descriptive; not causality)
    min_uses_detail = st.slider("Minimum uses (hotel anchors)", 2, 30, 6, 1)
    tag_perf = sub_hl.groupby("hashtag").agg(
        uses=("hashtag", "size"),
        median_ratio=("comment_like_ratio", "median"),
        avg_comments=("commentsCount", "mean"),
        theme=("hashtag_theme", lambda s: s.iloc[0] if len(s) else "Other")
    ).reset_index()
    tag_perf = tag_perf[tag_perf["uses"] >= min_uses_detail].copy()
    tag_perf = tag_perf.sort_values(["median_ratio", "uses"], ascending=[False, False]).head(20)

    d1, d2 = st.columns([1, 1])
    with d1:
        st.markdown("**Anchor tags + associated depth**")
        st.dataframe(
            tag_perf.round({"median_ratio": 4, "avg_comments": 2}),
            use_container_width=True,
            height=420
        )
        st.caption("Shown for descriptive anchoring; interpret as association, not a driver.")

    with d2:
        st.markdown("**Hashtag count distribution (signal discipline)**")
        dist = sub["hashtag_count"].value_counts().sort_index().reset_index()
        dist.columns = ["hashtag_count", "posts"]
        chart = alt.Chart(dist).mark_bar().encode(
            x=alt.X("hashtag_count:O", title="Hashtags per post"),
            y=alt.Y("posts:Q", title="Posts"),
            tooltip=["hashtag_count", "posts"],
        ).properties(height=420)
        st.altair_chart(chart, use_container_width=True)

st.divider()
st.markdown(
    """
    <div class="small-note">
      <b>Method note:</b> All “quality” signals here are derived only from likes and comment counts.
      No comment text, sentiment, demographics, or audience inference is used. Interpret results as <i>associations</i>, not causality.
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Optional: Streamlit theme polish
# -----------------------------
with st.expander("Optional: Make the UI feel more luxury (recommended)"):
    st.code(
        """Create a file at: .streamlit/config.toml

[theme]
base="dark"
primaryColor="#c8b37a"
backgroundColor="#0f0f0f"
secondaryBackgroundColor="#171717"
textColor="#f1f1f1"
font="serif"
""",
        language="toml"
    )
