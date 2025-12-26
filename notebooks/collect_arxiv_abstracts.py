import arxiv
import pandas as pd

# =========================
# AYARLAR
# =========================
SEARCH_QUERY = "machine learning"
MAX_RESULTS = 4000   # biraz fazla çekiyoruz, filtreleyeceğiz
OUTPUT_FILE = "arxiv_abstracts.csv"

# =========================
# ARXIV SEARCH
# =========================
search = arxiv.Search(
    query=SEARCH_QUERY,
    max_results=MAX_RESULTS,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

records = []

print("📥 arXiv abstract'ler çekiliyor...")

for result in search.results():
    abstract = result.summary.replace("\n", " ").strip()

    # Çok kısa abstract'leri alma
    if len(abstract) < 400:
        continue

    records.append({
        "arxiv_id": result.entry_id,
        "title": result.title,
        "abstract": abstract
    })

print(f"✅ Toplanan geçerli abstract sayısı: {len(records)}")

# =========================
# CSV KAYDET
# =========================
df = pd.DataFrame(records)
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"💾 Kaydedildi → {OUTPUT_FILE}")
