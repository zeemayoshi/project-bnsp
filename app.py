"""
Dashboard Business Intelligence
Dampak Penggunaan AI Generatif terhadap Performa Akademik & Kesejahteraan Mahasiswa
=====================================================================================
SERTIFIKASI BNSP — DATA ANALYST
Proyek Sertifikasi Data Analyst (FR.IA.04A) — LSP TIK Global
Skema : TIKGLOBAL/LSP-0039/SKM/2024
Assessor : Dr. Septian Rahardiantoro, S.Stat, M.Si
Peserta  : Muhammad Agal Lulanika

Analisis end-to-end 50.000 catatan mahasiswa · Imputasi GPA ·
Segmentasi K-Means · Hierarchical · DBSCAN · Evidence-based AI policy.

Lima pertanyaan bisnis yang dijawab (Modul 1 — Identifikasi Kebutuhan Bisnis):
  Q1. Apakah AI menaikkan atau menurunkan IPK?
      → Target: Δ GPA (Post − Pre) vs intensitas AI
  Q2. Apakah ketergantungan AI menurunkan retensi?
      → Target: Skill_Retention vs AI Dependency
  Q3. Bagaimana kebijakan kampus memengaruhi hasil?
      → Target: Burnout, Anxiety, GPA per kebijakan
  Q4. Segmen mahasiswa mana yang paling berisiko?
      → Target: K-Means · Hierarchical · DBSCAN

Stakeholder: Divisi Riset & Kebijakan · Rektorat · Dosen & Penjaminan Mutu ·
             Unit Konseling · Mahasiswa

Cara menjalankan:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------------------------------- #
# 0. KONFIGURASI HALAMAN
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="BI Dashboard — Dampak AI Generatif pada Mahasiswa | BNSP Data Analyst",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Palet warna konsisten
PRIMARY = "#2563EB"
ACCENT = "#F59E0B"
GOOD = "#16A34A"
BAD = "#DC2626"
SEQ = px.colors.sequential.Blues
POLICY_ORDER = ["Strict_Ban", "Allowed_With_Citation", "Actively_Encouraged"]
BURNOUT_ORDER = ["Low", "Medium", "High"]
SKILL_ORDER = ["Beginner", "Intermediate", "Advanced"]
YEAR_ORDER = ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"]

st.markdown(
    """
    <style>
    .main > div {padding-top: 1.2rem;}
    [data-testid="stMetricValue"] {font-size: 1.6rem;}
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    h1 {color:#1E3A8A;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# 1. PEMUATAN & PEMBERSIHAN DATA  (validasi kualitas data)
# --------------------------------------------------------------------------- #
def _find_csv():
    """Cari dataset di beberapa lokasi/nama umum agar app portabel.

    Menerima dataset mentah ATAU dataset bersih (clean). Mengembalikan path
    bila ditemukan, atau None bila tidak ada (akan ditangani dengan uploader).
    """
    try:
        here = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        here = os.getcwd()
    names = ["ai_student_impact_dataset.csv", "ai_student_impact_clean.csv"]
    dirs = [".", here, "data", "/mnt/user-data/uploads"]
    for d in dirs:
        for nm in names:
            p = os.path.join(d, nm)
            if os.path.exists(p):
                return p
    return None


# Batas validasi sesuai metadata FR.IA.04A
VALID_RANGES = {
    "Pre_Semester_GPA": (1.18, 4.00),
    "Post_Semester_GPA": (1.00, 4.00),
    "Weekly_GenAI_Hours": (0, 40),
    "Tool_Diversity": (1, 5),
    "Traditional_Study_Hours": (1, 36),
    "Perceived_AI_Dependency": (1, 10),
    "Anxiety_Level_During_Exams": (1, 10),
    "Skill_Retention_Score": (0, 100),
}
VALID_YEARS = set(YEAR_ORDER)


@st.cache_data(show_spinner="Memuat & membersihkan data…")
def load_data(source):
    """Return (df_clean, quality_report). `source` = path CSV atau bytes."""
    raw = pd.read_csv(source)
    n0 = len(raw)
    report = {"rows_initial": n0, "issues": []}

    df = raw.copy()

    # -- a. Missing values --------------------------------------------------
    miss_rows = df.drop(columns=["Student_ID"]).isnull().all(axis=1).sum()
    miss_cells = int(df.isnull().sum().sum())
    report["issues"].append(
        ("Missing values", f"{miss_cells} sel kosong; {miss_rows} baris kosong total",
         "Baris kosong total dihapus; tidak ada imputasi dipaksakan")
    )
    # hapus baris yang seluruh atribut (selain ID) kosong
    df = df[~df.drop(columns=["Student_ID"]).isnull().all(axis=1)].copy()

    # -- b. Inkonsistensi kategori -----------------------------------------
    bad_year = (~df["Year_of_Study"].isin(VALID_YEARS) & df["Year_of_Study"].notna()).sum()
    if bad_year:
        report["issues"].append(
            ("Inkonsistensi kategori", f"{bad_year} nilai Year_of_Study invalid (mis. '123')",
             "Nilai invalid diset NaN lalu baris di-drop")
        )
    df.loc[~df["Year_of_Study"].isin(VALID_YEARS), "Year_of_Study"] = np.nan

    # -- c. Outlier / nilai di luar rentang valid --------------------------
    out_total = 0
    for col, (lo, hi) in VALID_RANGES.items():
        mask = (df[col] < lo) | (df[col] > hi)
        n = int(mask.sum())
        if n:
            out_total += n
            report["issues"].append(
                ("Outlier / out-of-range", f"{col}: {n} nilai di luar [{lo}, {hi}]",
                 "Nilai di luar rentang diset NaN")
            )
            df.loc[mask, col] = np.nan

    # -- d. Konversi tipe ---------------------------------------------------
    df["Paid_Subscription"] = (
        df["Paid_Subscription"].map({True: True, False: False, "True": True, "False": False})
    )
    for c in ["Tool_Diversity", "Perceived_AI_Dependency", "Anxiety_Level_During_Exams"]:
        df[c] = df[c].round().astype("Int64")

    # -- e. Buang baris dengan target utama kosong --------------------------
    df = df.dropna(subset=["Post_Semester_GPA", "Pre_Semester_GPA", "Year_of_Study"])

    # Fitur turunan
    df["GPA_Delta"] = (df["Post_Semester_GPA"] - df["Pre_Semester_GPA"]).round(3)
    df["GPA_Improved"] = np.where(df["GPA_Delta"] >= 0, "Naik/Stabil", "Turun")
    bins = [-0.1, 2, 5, 10, 20, 40]
    labels = ["0-2 jam", "2-5 jam", "5-10 jam", "10-20 jam", "20-40 jam"]
    df["GenAI_Hours_Band"] = pd.cut(df["Weekly_GenAI_Hours"], bins=bins, labels=labels)

    report["rows_final"] = len(df)
    report["rows_removed"] = n0 - len(df)
    report["outliers_total"] = out_total
    return df, report


_csv_path = _find_csv()
if _csv_path is None:
    st.title("🎓 Dampak AI Generatif pada Mahasiswa")
    st.warning(
        "File dataset tidak ditemukan di folder ini. Letakkan "
        "`ai_student_impact_dataset.csv` **atau** `ai_student_impact_clean.csv` "
        "di folder yang sama dengan `app.py`, atau unggah di bawah ini."
    )
    up = st.file_uploader("Unggah dataset CSV", type=["csv"])
    if up is None:
        st.stop()
    df, QR = load_data(up.getvalue())
else:
    st.caption(f"📂 Sumber data: `{os.path.basename(_csv_path)}`")
    df, QR = load_data(_csv_path)

# --------------------------------------------------------------------------- #
# 2. SIDEBAR — FILTER & DRILL-DOWN
# --------------------------------------------------------------------------- #
st.sidebar.markdown("## 🎛️ Filter Data")
st.sidebar.caption("Semua visualisasi & KPI bereaksi terhadap filter ini.")


def msel(label, col, order=None):
    opts = [o for o in (order or sorted(df[col].dropna().unique())) if o in df[col].unique()]
    return st.sidebar.multiselect(label, opts, default=opts)


f_major = msel("Bidang Studi", "Major_Category")
f_year = msel("Jenjang", "Year_of_Study", YEAR_ORDER)
f_policy = msel("Kebijakan Institusi", "Institutional_Policy", POLICY_ORDER)
f_skill = msel("Prompt Engineering", "Prompt_Engineering_Skill", SKILL_ORDER)
f_burn = msel("Risiko Burnout", "Burnout_Risk_Level", BURNOUT_ORDER)
f_paid = st.sidebar.radio("Langganan Berbayar", ["Semua", "Berbayar", "Gratis"], horizontal=True)

hmin, hmax = float(df["Weekly_GenAI_Hours"].min()), float(df["Weekly_GenAI_Hours"].max())
f_hours = st.sidebar.slider("Jam AI / minggu", hmin, hmax, (hmin, hmax))
dep_lo, dep_hi = st.sidebar.slider("Skor Ketergantungan AI", 1, 10, (1, 10))

# Terapkan filter
mask = (
    df["Major_Category"].isin(f_major)
    & df["Year_of_Study"].isin(f_year)
    & df["Institutional_Policy"].isin(f_policy)
    & df["Prompt_Engineering_Skill"].isin(f_skill)
    & df["Burnout_Risk_Level"].isin(f_burn)
    & df["Weekly_GenAI_Hours"].between(*f_hours)
    & df["Perceived_AI_Dependency"].between(dep_lo, dep_hi)
)
if f_paid == "Berbayar":
    mask &= df["Paid_Subscription"] == True
elif f_paid == "Gratis":
    mask &= df["Paid_Subscription"] == False

d = df[mask].copy()
st.sidebar.markdown("---")
st.sidebar.metric("Mahasiswa terpilih", f"{len(d):,}", f"{len(d)/len(df)*100:.1f}% dari total")
if st.sidebar.button("🔄 Reset (muat ulang)"):
    st.rerun()

# --------------------------------------------------------------------------- #
# 3. HEADER & KPI
# --------------------------------------------------------------------------- #
st.title("🎓 Dampak Penggunaan AI Generatif terhadap Performa Akademik & Kesejahteraan Mahasiswa")
st.markdown(
    "**Business Intelligence Dashboard** — Sertifikasi BNSP · Data Analyst · "
    "mendukung perumusan kebijakan AI berbasis bukti (*evidence-based AI policy*).  \n"
    f"Analisis end-to-end · {QR['rows_final']:,} catatan bersih dari {QR['rows_initial']:,} mentah · "
    "16 variabel · Imputasi GPA · Segmentasi **K-Means · Hierarchical · DBSCAN**."
)
st.caption(
    "📋 Skema: TIKGLOBAL/LSP-0039/SKM/2024 · FR.IA.04A · LSP TIK Global  |  "
    "👩‍🏫 Assessor: Dr. Septian Rahardiantoro, S.Stat, M.Si  |  "
    "🎓 Peserta: Muhammad Agal Lulanika"
)

if len(d) == 0:
    st.warning("Tidak ada data yang cocok dengan filter. Longgarkan filter di sidebar.")
    st.stop()

k = st.columns(5)
k[0].metric("Rata-rata Post GPA", f"{d['Post_Semester_GPA'].mean():.2f}",
            f"{d['GPA_Delta'].mean():+.3f} vs awal",
            help="Slide 5: Median IPK 3,21 → 3,42 · rata-rata kenaikan +0,20")
k[1].metric("Skill Retention", f"{d['Skill_Retention_Score'].mean():.1f}",
            help="Skor retensi keterampilan pasca-semester (0–100). Moderate user: 77 (tertinggi)")
k[2].metric("Jam AI / minggu", f"{d['Weekly_GenAI_Hours'].mean():.1f}",
            help="Sweet spot ~12 jam/minggu. Heavy user (>12 jam): retensi turun ke 72,7")
k[3].metric("Risiko Burnout High",
            f"{(d['Burnout_Risk_Level'].eq('High').mean()*100):.1f}%",
            help="Heavy user: 6,2× lebih tinggi burnout dibanding Light user (Slide 6)")
k[4].metric("Kecemasan Ujian", f"{d['Anxiety_Level_During_Exams'].mean():.1f}/10",
            help="Strict_Ban: 4,89 vs ±4,12 kebijakan lain (Slide 7)")

st.markdown("---")

# --------------------------------------------------------------------------- #
# 4. TABS
# --------------------------------------------------------------------------- #
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Ringkasan", "🤖 AI vs Akademik", "🧠 Kesejahteraan & Burnout",
     "🏛️ Kebijakan & Profil", "🔍 Kualitas Data"]
)

# ===== TAB 1: RINGKASAN ==================================================== #
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Distribusi GPA Akhir per Bidang Studi")
        fig = px.box(d, x="Major_Category", y="Post_Semester_GPA",
                     color="Major_Category", points=False,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Post GPA",
                          height=380)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Komposisi Risiko Burnout")
        bd = (d["Burnout_Risk_Level"].value_counts()
              .reindex(BURNOUT_ORDER).reset_index())
        bd.columns = ["Burnout", "Jumlah"]
        fig = px.pie(bd, names="Burnout", values="Jumlah", hole=0.5,
                     color="Burnout",
                     color_discrete_map={"Low": GOOD, "Medium": ACCENT, "High": BAD})
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Perubahan GPA (Awal → Akhir) per Jenjang & Bidang")
    pv = (d.groupby(["Year_of_Study", "Major_Category"])["GPA_Delta"]
          .mean().reset_index())
    pv["Year_of_Study"] = pd.Categorical(pv["Year_of_Study"], YEAR_ORDER, ordered=True)
    pv = pv.sort_values("Year_of_Study")
    fig = px.density_heatmap(
        pv, x="Year_of_Study", y="Major_Category", z="GPA_Delta",
        color_continuous_scale="RdYlGn", text_auto=".3f")
    fig.update_layout(height=360, xaxis_title="", yaxis_title="",
                      coloraxis_colorbar_title="Δ GPA")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Hijau = peningkatan GPA; merah = penurunan. Hover untuk detail.  "
        "**Temuan 2 (Slide 5):** Median IPK naik 3,21 → 3,42 · rata-rata kenaikan **+0,20** · "
        "87,6% mahasiswa membaik · STEM: kenaikan median terbesar (+0,23)."
    )

# ===== TAB 2: AI vs AKADEMIK ============================================== #
with tab2:
    st.markdown(
        "#### Pertanyaan: *Apakah lebih banyak jam AI = hasil lebih baik?*  \n"
        "> **Temuan 1 — Efek \"Sweet Spot\" (Slide 4):** Manfaat akademik bukan soal banyaknya jam AI, "
        "tapi takarannya. Kenaikan IPK **memuncak pada penggunaan moderat (~12 jam/minggu)**: "
        "+0,227 IPK · retensi 77 (tertinggi). Heavy user: hanya +0,173 IPK · retensi turun ke 72,7. "
        "Korelasi linear jam AI ↔ IPK ≈ 0 — **polanya, bukan volumenya** yang menentukan."
    )
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Jam AI vs Skill Retention")
        band = (d.groupby("GenAI_Hours_Band", observed=True)
                .agg(Retention=("Skill_Retention_Score", "mean"),
                     PostGPA=("Post_Semester_GPA", "mean"),
                     n=("Student_ID", "count")).reset_index())
        fig = px.bar(band, x="GenAI_Hours_Band", y="Retention",
                     color="Retention", color_continuous_scale="Blues_r",
                     text="Retention")
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_layout(height=380, xaxis_title="Jam AI / minggu",
                          yaxis_title="Rata-rata Skill Retention",
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Ketergantungan AI vs Retention")
        dep = (d.groupby("Perceived_AI_Dependency")
               .agg(Retention=("Skill_Retention_Score", "mean"),
                    Anxiety=("Anxiety_Level_During_Exams", "mean")).reset_index())
        fig = go.Figure()
        fig.add_bar(x=dep["Perceived_AI_Dependency"], y=dep["Retention"],
                    name="Skill Retention", marker_color=PRIMARY)
        fig.add_scatter(x=dep["Perceived_AI_Dependency"], y=dep["Anxiety"] * 10,
                        name="Kecemasan (×10)", mode="lines+markers",
                        marker_color=BAD, yaxis="y2")
        fig.update_layout(
            height=380, xaxis_title="Skor Ketergantungan Ai (1–10)",
            yaxis_title="Skill Retention",
            yaxis2=dict(title="Kecemasan ×10", overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Use Case AI: Performa vs Retensi")
    uc = (d.groupby("Primary_Use_Case")
          .agg(PostGPA=("Post_Semester_GPA", "mean"),
               Retention=("Skill_Retention_Score", "mean"),
               Dependency=("Perceived_AI_Dependency", "mean"),
               n=("Student_ID", "count")).reset_index())
    fig = px.scatter(uc, x="PostGPA", y="Retention", size="n", color="Dependency",
                     text="Primary_Use_Case", color_continuous_scale="OrRd",
                     size_max=55)
    fig.update_traces(textposition="top center")
    fig.update_layout(height=440, xaxis_title="Rata-rata Post GPA",
                      yaxis_title="Rata-rata Skill Retention",
                      coloraxis_colorbar_title="Dependency")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Ukuran titik = jumlah mahasiswa; warna = tingkat ketergantungan AI. "
        "Use case *Direct_Answer_Generation* cenderung dependency tinggi.  "
        "**Slide 4:** Korelasi linear jam AI ↔ IPK ≈ 0 — pola penggunaan, bukan volume, yang menentukan."
    )

    # Matriks korelasi
    st.subheader("Matriks Korelasi (variabel numerik)")
    num = ["Weekly_GenAI_Hours", "Perceived_AI_Dependency", "Traditional_Study_Hours",
           "Anxiety_Level_During_Exams", "Skill_Retention_Score",
           "Pre_Semester_GPA", "Post_Semester_GPA", "GPA_Delta"]
    corr = d[num].astype(float).corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)

# ===== TAB 3: KESEJAHTERAAN =============================================== #
with tab3:
    st.markdown(
        "> **Temuan 3 — Intensitas Berlebih Memicu Burnout (Slide 6):** "
        "Proporsi mahasiswa berisiko burnout TINGGI naik tajam seiring intensitas AI. "
        "Heavy user: **6,2× lebih tinggi** risiko burnout dibanding Light user. "
        "Penggunaan berat (>12 jam/minggu) sebaiknya menjadi pemicu deteksi dini & rujukan konseling."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Burnout vs Jam AI & Belajar Tradisional")
        bb = (d.groupby("Burnout_Risk_Level")
              .agg(AI=("Weekly_GenAI_Hours", "mean"),
                   Trad=("Traditional_Study_Hours", "mean"),
                   Anx=("Anxiety_Level_During_Exams", "mean")).reindex(BURNOUT_ORDER).reset_index())
        fig = go.Figure()
        fig.add_bar(x=bb["Burnout_Risk_Level"], y=bb["AI"], name="Jam AI", marker_color=ACCENT)
        fig.add_bar(x=bb["Burnout_Risk_Level"], y=bb["Trad"], name="Jam Belajar Tradisional",
                    marker_color=PRIMARY)
        fig.update_layout(barmode="group", height=380, yaxis_title="Jam / minggu")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Kecemasan Ujian per Bidang Studi")
        fig = px.violin(d, x="Major_Category", y="Anxiety_Level_During_Exams",
                        color="Major_Category", box=True, points=False,
                        color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(showlegend=False, height=380, xaxis_title="",
                          yaxis_title="Skor Kecemasan")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Peta Risiko: Ketergantungan AI × Jam AI → Tingkat Burnout")
    hb = (d.groupby(["Perceived_AI_Dependency", "GenAI_Hours_Band"], observed=True)
          .apply(lambda x: (x["Burnout_Risk_Level"].eq("High").mean() * 100), include_groups=False)
          .reset_index(name="Pct_High_Burnout"))
    fig = px.density_heatmap(
        hb, x="GenAI_Hours_Band", y="Perceived_AI_Dependency", z="Pct_High_Burnout",
        color_continuous_scale="Reds", text_auto=".0f")
    fig.update_layout(height=420, xaxis_title="Jam AI / minggu",
                      yaxis_title="Ketergantungan AI",
                      coloraxis_colorbar_title="% Burnout High")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Persentase mahasiswa berisiko burnout tinggi pada tiap kombinasi.  "
        "**Slide 6:** Heavy user (>12 jam/mgg) → risiko burnout 6,2× dibanding Light user. "
        "Segmen C2 (≈22 jam AI/mgg): burnout 60,9% — prioritas intervensi & konseling."
    )

# ===== TAB 4: KEBIJAKAN =================================================== #
with tab4:
    st.markdown(
        "#### Pertanyaan: *Apakah kebijakan kampus memengaruhi hasil mahasiswa?*  \n"
        "> **Temuan 4 — Pelarangan Total Justru Kontraproduktif (Slide 7):**  \n"
        "| Kebijakan | Burnout Tinggi | Retensi |  \n"
        "|---|---|---|  \n"
        "| 🚫 Strict_Ban (larangan total) | **29,8%** (tertinggi) | 74,99 (terendah) |  \n"
        "| 📝 Allowed_With_Citation | 23,8% | 75,91 |  \n"
        "| ✅ Actively_Encouraged | 23,8% | **76,13** (tertinggi) |  \n\n"
        "Kecemasan ujian pada *Strict_Ban*: **4,89** vs ±4,12 pada kebijakan lain — "
        "pelarangan menaikkan tekanan tanpa manfaat akademik."
    )
    pol = (d.groupby("Institutional_Policy")
           .agg(PostGPA=("Post_Semester_GPA", "mean"),
                Retention=("Skill_Retention_Score", "mean"),
                Dependency=("Perceived_AI_Dependency", "mean"),
                HighBurnout=("Burnout_Risk_Level", lambda x: x.eq("High").mean() * 100),
                n=("Student_ID", "count"))
           .reindex(POLICY_ORDER).reset_index())

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Skill Retention per Kebijakan")
        fig = px.bar(pol, x="Institutional_Policy", y="Retention", color="Institutional_Policy",
                     text="Retention", color_discrete_sequence=px.colors.qualitative.Prism)
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_layout(showlegend=False, height=380, xaxis_title="",
                          yaxis_title="Skill Retention")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("% Burnout High per Kebijakan")
        fig = px.bar(pol, x="Institutional_Policy", y="HighBurnout",
                     color="Institutional_Policy", text="HighBurnout",
                     color_discrete_sequence=px.colors.qualitative.Prism)
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(showlegend=False, height=380, xaxis_title="",
                          yaxis_title="% Burnout High")
        st.plotly_chart(fig, use_container_width=True)

    # Segmentasi K-Means dari slide 8
    st.subheader("Temuan 5 — Segmentasi: K-Means · Hierarchical · DBSCAN (Slide 8)")
    st.markdown(
        "Tiga algoritma saling melengkapi — **partisi, verifikasi struktur, dan deteksi pencilan** over-reliance.\n\n"
        "- **K-Means**: k=3 optimal (silhouette 0,184) — 3 segmen jelas\n"
        "- **Hierarchical (Ward)**: struktur 3 klaster konsisten (silhouette 0,14)\n"
        "- **DBSCAN**: 1 klaster inti + 1.500 noise (3,0%) pencilan ekstrem → peringatan dini over-reliance"
    )
    seg_data = {
        "Segmen (K-Means)": [
            "C0 · Hemat-AI, IPK tinggi",
            "C1 · Hemat-AI, IPK rendah",
            "C2 · Intensif, ketergantungan ⚠️",
        ],
        "n": [25421, 15711, 8863],
        "Jam AI/mgg": [5.4, 5.8, 21.7],
        "Retensi": [78.8, 73.3, 71.6],
        "Post GPA": [3.67, 2.81, 3.38],
        "Burnout High": ["14,7%", "21,3%", "60,9% 🚨"],
    }
    seg_df = pd.DataFrame(seg_data)
    st.dataframe(seg_df, use_container_width=True, hide_index=True)
    st.caption(
        "**C2 = prioritas intervensi**: ≈22 jam AI/mgg · ketergantungan tertinggi · burnout 60,9%.  "
        "DBSCAN noise = over-reliance ekstrem (≈29 jam AI, ~80% burnout) → sistem peringatan dini."
    )

    st.subheader("Profil 'Penggunaan AI Sehat' — Prompt Skill × Diversitas Tool")
    prof = (d.groupby(["Prompt_Engineering_Skill", "Tool_Diversity"])
            .agg(Retention=("Skill_Retention_Score", "mean"),
                 n=("Student_ID", "count")).reset_index())
    prof["Prompt_Engineering_Skill"] = pd.Categorical(
        prof["Prompt_Engineering_Skill"], SKILL_ORDER, ordered=True)
    fig = px.scatter(prof, x="Tool_Diversity", y="Retention",
                     color="Prompt_Engineering_Skill", size="n", size_max=40,
                     category_orders={"Prompt_Engineering_Skill": SKILL_ORDER},
                     color_discrete_sequence=[BAD, ACCENT, GOOD])
    fig.update_layout(height=420, xaxis_title="Jumlah Tool AI Digunakan",
                      yaxis_title="Rata-rata Skill Retention",
                      legend_title="Prompt Skill")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Mahasiswa dengan prompt skill **Advanced** umumnya mempertahankan retensi lebih tinggi "
        "pada level penggunaan tool yang sama.  "
        "**Slide 9 Rek. 3:** Pelatihan literasi & prompt AI wajib bagi mahasiswa baru — "
        "KPI: skor prompt Advanced ↑ 25%."
    )

# ===== TAB 5: KUALITAS DATA =============================================== #
with tab5:
    st.subheader("Laporan Validasi & Pembersihan Data — Modul 2–4")
    st.markdown(
        "**Kerangka CRISP-DM** · Pipeline: *raw → validated → clean* (data lineage, DAMA-DMBOK).  \n"
        "Tidy data (Wickham 2014) · EDA (Tukey 1977) · "
        "⚙️ **Penanganan nilai hilang: IMPUTASI, bukan drop** — "
        "Pre_GPA & Post_GPA diimputasi median per (bidang × jenjang), fallback median global."
    )
    c = st.columns(4)
    c[0].metric("Baris mentah", f"{QR['rows_initial']:,}", help="50.000 catatan mahasiswa")
    c[1].metric("Baris bersih", f"{QR['rows_final']:,}", help="Slide 3: 49.996 baris bersih (drop 4)")
    c[2].metric("Baris dibuang", f"{QR['rows_removed']:,}", help="Slide 3: 2 baris rusak struktur di-drop")
    c[3].metric("Outlier ditangani", f"{QR['outliers_total']:,}", help="6 isu kualitas data ditangani (Slide 3)")

    rep = pd.DataFrame(QR["issues"],
                       columns=["Jenis Masalah", "Temuan", "Strategi Penanganan"])
    st.dataframe(rep, use_container_width=True, hide_index=True)

    st.markdown(
        """
        **Catatan validasi (sesuai Slide 3 — 6 isu kualitas data utama):**
        - `Pre_GPA = 10,39` (di luar [1,4]) → **Imputasi median per grup** (bidang × jenjang)
        - `Post_GPA = 12,25` (di luar [1,4]) → **Imputasi median per grup** (bidang × jenjang)
        - `Tool_Diversity = 18` (di luar [1,5]) → **Imputasi median**
        - `Year_of_Study = "123"` (invalid) → **Imputasi modus**
        - 2 baris rusak struktur → **Record kolaps → drop**
        - Tipe campuran (True/False) → **Konversi logical & factor**
        - Variabel target utama = **Post_Semester_GPA** & **Skill_Retention_Score** · 16 variabel total.
        - Granularitas analisis = **per mahasiswa** (1 baris = 1 mahasiswa per semester).
        """
    )

    st.subheader("Pratinjau Data Bersih (terfilter)")
    st.dataframe(d.head(200), use_container_width=True, height=320)
    st.download_button("⬇️ Unduh dataset bersih (CSV)",
                       d.to_csv(index=False).encode("utf-8"),
                       "ai_student_impact_clean.csv", "text/csv")

# --------------------------------------------------------------------------- #
# 5. INSIGHT & REKOMENDASI KEBIJAKAN (dinamis dari data terfilter)
# --------------------------------------------------------------------------- #
st.markdown("---")
st.header("💡 Temuan Utama & Rekomendasi Kebijakan")
st.markdown("*Modul 7 — Rekomendasi Kebijakan Actionable (Slide 9)*")

corr_hours = d["Weekly_GenAI_Hours"].corr(d["Skill_Retention_Score"])
corr_dep = d["Perceived_AI_Dependency"].corr(d["Skill_Retention_Score"])
corr_trad = d["Traditional_Study_Hours"].corr(d["Skill_Retention_Score"])

cA, cB = st.columns(2)
with cA:
    st.markdown(
        f"""
        **5 Temuan Utama (dari Slide 4–8):**

        1. 🎯 **Sweet Spot AI (Slide 4):** Kenaikan IPK memuncak pada ~12 jam/minggu
           (Moderate: +0,227 IPK · retensi 77). Heavy user: hanya +0,173 IPK · retensi 72,7.
           Korelasi linear jam AI ↔ IPK ≈ 0 — **polanya, bukan volumenya**.

        2. 📈 **GPA Naik Merata (Slide 5):** Median IPK 3,21 → 3,42 · rata-rata **+0,20** ·
           87,6% mahasiswa membaik · STEM: kenaikan median terbesar (+0,23).

        3. 🔥 **Burnout Heavy User (Slide 6):** Risiko burnout **6,2× lebih tinggi** dibanding
           Light user. Penggunaan >12 jam/mgg = pemicu deteksi dini & konseling.

        4. 🚫 **Strict_Ban Kontraproduktif (Slide 7):** Burnout 29,8% (tertinggi) · retensi 74,99
           (terendah) · kecemasan ujian 4,89 vs ±4,12 kebijakan lain. Tanpa keunggulan akademik.

        5. 👥 **Segmentasi (Slide 8):** K-Means k=3 optimal (silhouette 0,184). C2 (≈22 jam AI,
           burnout 60,9%) = prioritas intervensi. DBSCAN: 1.500 noise (3%) = over-reliance ekstrem.

        *Korelasi data terfilter:*
        - Jam AI ↔ Retensi: **{corr_hours:+.2f}** · Dependency ↔ Retensi: **{corr_dep:+.2f}**
        - Belajar tradisional ↔ Retensi: **{corr_trad:+.2f}** (pendorong positif utama)
        """
    )
with cB:
    st.markdown(
        """
        **4 Rekomendasi Kebijakan Actionable (Slide 9):**

        1. 📝 **Kebijakan "Izinkan + Takar"**
           Adopsi *Allowed_With_Citation* dengan panduan penggunaan moderat — bukan pelarangan
           total. Strict_Ban terbukti menaikkan burnout & kecemasan tanpa manfaat akademik.
           > 🎯 KPI: % mahasiswa zona moderat ↑

        2. 🚨 **Sistem Peringatan Dini Burnout**
           Skrining otomatis segmen **C2** & noise DBSCAN (dependency tinggi), rujuk konseling.
           Target: heavy user >12 jam/mgg atau dependency ≥7.
           > 🎯 KPI: % burnout tinggi < 20% per 2 semester

        3. 🧠 **Pelatihan Literasi & Prompt AI**
           Wajib bagi mahasiswa baru. Keterampilan prompt menentukan retensi — bukan durasi.
           Prompt skill Advanced → retensi lebih tinggi pada intensitas penggunaan yang sama.
           > 🎯 KPI: skor prompt Advanced ↑ 25%

        4. 📚 **Desain Ulang Asesmen**
           Nilai proses & penalaran; jaga keseimbangan jam belajar tradisional.
           Hindari tugas yang mendorong *Direct_Answer_Generation*.
           > 🎯 KPI: korelasi dependency–retensi → 0

        ---
        **Kesimpulan (Slide 10):** *Bukan kuantitas, melainkan pola penggunaan AI yang sehat
        dan terpandu yang menentukan keberhasilan akademik sekaligus kesejahteraan mahasiswa.*
        """
    )

st.caption(
    "📋 Proyek Sertifikasi Data Analyst · FR.IA.04A · LSP TIK Global · Skema: TIKGLOBAL/LSP-0039/SKM/2024  |  "
    "7 Unit Kompetensi: Kebutuhan Data · Mengumpulkan Data · Menelaah Data · Memvalidasi Data · "
    "Objek Data · Business Intelligence · Laporan Analisis  |  "
    "👩‍🏫 Assessor: Dr. Septian Rahardiantoro, S.Stat, M.Si  |  🎓 Peserta: Muhammad Agal Lulanika"
)
