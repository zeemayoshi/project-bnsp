# 🎓 BI Dashboard — Dampak AI Generatif pada Mahasiswa

Dashboard Business Intelligence untuk proyek Sertifikasi **Data Analyst**
(FR.IA.04A · LSP TIK Global · Skema TIKGLOBAL/LSP-0039/SKM/2024).

## Pertanyaan bisnis yang dijawab
1. Apakah penggunaan AI generatif meningkatkan/menurunkan performa akademik (Post GPA)?
2. Bagaimana AI memengaruhi retensi keterampilan (Skill Retention)?
3. Faktor pendorong risiko burnout & kecemasan ujian?
4. Apakah kebijakan institusi berdampak pada hasil mahasiswa?
5. Profil penggunaan AI yang "sehat"?

## Fitur
- **5 tab analitik**: Ringkasan, AI vs Akademik, Kesejahteraan & Burnout, Kebijakan & Profil, Kualitas Data.
- **Filter & drill-down** interaktif (bidang, jenjang, kebijakan, prompt skill, burnout, slider jam AI & ketergantungan, status langganan).
- **>10 visualisasi interaktif** Plotly (box, donut, heatmap, scatter bubble, korelasi, violin, dual-axis).
- **Pipeline validasi data** otomatis: missing values, outlier/out-of-range, inkonsistensi kategori — dengan laporan & unduh dataset bersih.
- **Insight & ≥4 rekomendasi kebijakan** dihitung dinamis dari data terfilter.

## Cara menjalankan
```bash
pip install -r requirements.txt
# letakkan ai_student_impact_dataset.csv di folder yang sama dengan app.py
streamlit run app.py
```
Buka http://localhost:8501 di browser.

## Struktur file
```
app.py                         # aplikasi Streamlit
requirements.txt               # dependensi
ai_student_impact_dataset.csv  # dataset mentah (50.000 baris)
ai_student_impact_clean.csv    # dataset bersih hasil validasi (opsional)
```
