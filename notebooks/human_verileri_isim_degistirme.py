import pandas as pd

INPUT_FILE = "arxiv_abstracts.csv"
OUTPUT_FILE = "human_3000.csv"

df = pd.read_csv(INPUT_FILE)

# sadece abstract al
df = df[["abstract"]].dropna()

# rastgele 3000 tane seç
df = df.sample(n=3000, random_state=42)

# kolon ismini değiştir
df.rename(columns={"abstract": "text"}, inplace=True)

# label ekle (human = 0)
df["label"] = 0

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("✅ human_3000.csv oluşturuldu")
print(df.head())
