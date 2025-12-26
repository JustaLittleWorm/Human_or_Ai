import pandas as pd

human = pd.read_csv("human_3000.csv")
human["label"] = 0

ai = pd.read_csv("ai_3000_clean.csv")
ai["label"] = 1

df = pd.concat([human, ai], ignore_index=True)

df.shape
df["label"].value_counts()
df.to_csv(
    "ai_human_6000_final.csv",
    index=False,
    encoding="utf-8-sig"
)