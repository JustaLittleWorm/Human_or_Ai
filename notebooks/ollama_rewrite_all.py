import pandas as pd
import subprocess
import time

OLLAMA_PATH = r"C:\Users\burak\AppData\Local\Programs\Ollama\ollama.exe"
MODEL = "llama3:8b"

INPUT_FILE = "human_3000.csv"
OUTPUT_FILE = "ai_3000.csv"

PROMPT_TEMPLATE = """
Rewrite the following academic text in a formal, neutral, and technically precise academic style.

Rules:
- Output ONLY the rewritten text.
- Do NOT include introductions, explanations, or meta phrases.
- Do NOT add titles, headings, or section labels.
- Do NOT add or remove information.
- Keep approximately the same length.

Text: \"\"\" {text} \"\"\" """

df = pd.read_csv(INPUT_FILE)

ai_texts = []
try:
    for idx, row in df.iterrows():
        prompt = PROMPT_TEMPLATE.format(text=row["text"])

        result = subprocess.run(
            [OLLAMA_PATH, "run", MODEL],
            input=prompt,
            text=True,
            capture_output=True
        )

        rewritten = result.stdout.strip()

        if not rewritten or len(rewritten) < 50:
            rewritten = row["text"]  # fallback (çok nadir olur)

        ai_texts.append(rewritten)

        if (idx + 1) % 25 == 0:
            print(f"➡ {idx+1}/{len(df)} AI rewrite tamamlandı")
            time.sleep(0.3)  # sistemi yormasın

except KeyboardInterrupt:
    print("⛔ Manuel durdurma algılandı")

finally:
    print("💾 CSV kaydediliyor...")
    df_ai = pd.DataFrame({
        "text": ai_texts,
        "label": 1
    })
    df_ai.to_csv("ai_partial.csv", index=False, encoding="utf-8-sig")
    print("✅ Güvenli şekilde kaydedildi")



df_ai = pd.DataFrame({
    "text": ai_texts,
    "label": 1
})

df_ai.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("✅ ai_verileri.csv başarıyla oluşturuldu")
