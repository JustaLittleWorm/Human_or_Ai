import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ai_3000_clean.csv")

print(df.shape)
df.head()
print(df.columns)
print(df["label"].value_counts())
print("Boş satır:", df.isnull().sum())

df.dropna(subset=["text"], inplace=True)

before = len(df)
df.drop_duplicates(subset=["text"], inplace=True)
after = len(df)

print(f"Silinen duplicate: {before - after}")
df["char_len"] = df["text"].str.len()
df["word_len"] = df["text"].str.split().str.len()

df[["char_len", "word_len"]].describe()


df["char_len"].hist(bins=50)
plt.title("Karakter Uzunluğu Dağılımı")
plt.show()
