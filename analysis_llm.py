import pandas as pd
from openai import OpenAI
import os

# ==============================
# SETUP
# ==============================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

df = pd.read_csv("hotel_posts_themes_ready.csv")

# We will classify into the SAME themes for consistency
THEMES = ["Brand", "Experience", "Seasonal", "Other"]

# ==============================
# LLM CLASSIFICATION FUNCTION
# ==============================
def classify_with_llm(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return "Other"

    prompt = f"""
You are classifying Instagram posts from luxury hotels.

Themes:
- Brand: brand identity, hotel name, prestige
- Experience: spa, dining, rooms, service, wellness
- Seasonal: holidays, seasons, events
- Other

Text:
{text}

Respond with ONLY one word from:
Brand, Experience, Seasonal, Other
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()

# ==============================
# RUN LLM ON SAMPLE (IMPORTANT)
# ==============================
# ⚠️ Do NOT run on entire dataset at once (cost + speed)
sample_df = df.sample(300, random_state=42)

sample_df["theme_llm"] = sample_df["text_for_llm"].apply(classify_with_llm)

sample_df.to_csv("hotel_posts_llm_classified.csv", index=False)

print("✅ LLM classification complete")
