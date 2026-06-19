import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(
    page_title="NumeraBoard",
    page_icon="📊",
    layout="wide"
)

df_awal = pd.read_csv("numeraboard_data.csv")
df_awal.columns = df_awal.columns.str.strip().str.lower()

model = joblib.load("random_forest_model.pkl")

if "nama siswa" not in df_awal.columns:
    df_awal.insert(0, "nama siswa", ["Siswa " + str(i + 1) for i in range(len(df_awal))])

if "kategori_numerasi" not in df_awal.columns and "prediksi_random_forest" in df_awal.columns:
    df_awal["kategori_numerasi"] = df_awal["prediksi_random_forest"]

def buat_rekomendasi(kategori):
    if kategori == "Rendah":
        return "Remedial"
    elif kategori == "Sedang":
        return "Penguatan Materi"
    elif kategori == "Tinggi":
        return "Pengayaan"
    else:
        return "Belum Diprediksi"

if "rekomendasi" not in df_awal.columns:
    df_awal["rekomendasi"] = df_awal["kategori_numerasi"].apply(buat_rekomendasi)

if "cluster" not in df_awal.columns:
    df_awal["cluster"] = df_awal["kategori_numerasi"].map({
        "Rendah": "Cluster Rendah",
        "Sedang": "Cluster Sedang",
        "Tinggi": "Cluster Tinggi"
    })

if "data_baru" not in st.session_state:
    st.session_state.data_baru = pd.DataFrame(
        columns=[
            "nama siswa",
            "math score",
            "reading score",
            "writing score",
            "cluster",
            "kategori_numerasi",
            "rekomendasi"
        ]
    )

df = pd.concat([df_awal, st.session_state.data_baru], ignore_index=True)

st.markdown("""
<style>
[data-testid="stSidebar"]{
    background: linear-gradient(180deg,#3B5BDB,#7B2CBF);
}
[data-testid="stSidebar"] *{
    color:white;
}
.block-container{
    padding-top: 2rem;
}
.card{
    border-radius:22px;
    padding:24px;
    color:white;
    text-align:center;
    box-shadow:0px 8px 20px rgba(0,0,0,0.15);
}
.blue{background:linear-gradient(135deg,#00B4D8,#4361EE);}
.purple{background:linear-gradient(135deg,#9D4EDD,#C77DFF);}
.pink{background:linear-gradient(135deg,#FF4D6D,#F72585);}
.orange{background:linear-gradient(135deg,#FF9E00,#FF6D00);}
.green{background:linear-gradient(135deg,#2DC653,#00B894);}
.big-number{
    font-size:38px;
    font-weight:800;
}
.white-box{
    background:white;
    border-radius:18px;
    padding:22px;
    box-shadow:0px 6px 18px rgba(0,0,0,0.08);
    margin-bottom:18px;
}
.rekom-card{
    background:white;
    border-left:6px solid #4361EE;
    border-radius:16px;
    padding:18px;
    box-shadow:0px 6px 18px rgba(0,0,0,0.08);
    margin-bottom:16px;
}
.badge{
    padding:5px 12px;
    border-radius:20px;
    color:white;
    font-size:13px;
    font-weight:700;
}
.badge-rendah{background:#F72585;}
.badge-sedang{background:#FFB703;}
.badge-tinggi{background:#2DC653;}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("📊 NumeraBoard")
st.sidebar.caption("Monitoring Numerasi Siswa")

menu = st.sidebar.radio(
    "Pilih Menu",
    [
        "🏠 Dashboard Utama",
        "📝 Input Data Siswa",
        "🤖 Prediksi Numerasi",
        "📈 Monitoring Perkembangan",
        "💡 Rekomendasi Pembelajaran"
    ]
)

if menu == "🏠 Dashboard Utama":

    st.title("NumeraBoard")
    st.caption("Dashboard Monitoring Kemampuan Numerasi Siswa")

    total = len(df)
    rata2 = round(df["math score"].astype(float).mean(), 1)
    tinggi = len(df[df["kategori_numerasi"] == "Tinggi"])
    rendah = len(df[df["kategori_numerasi"] == "Rendah"])

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"<div class='card blue'><h4>👨‍🎓 Total Siswa</h4><div class='big-number'>{total}</div></div>", unsafe_allow_html=True)

    with c2:
        st.markdown(f"<div class='card purple'><h4>📚 Rata-rata</h4><div class='big-number'>{rata2}</div></div>", unsafe_allow_html=True)

    with c3:
        st.markdown(f"<div class='card pink'><h4>⭐ Kategori Tinggi</h4><div class='big-number'>{tinggi}</div></div>", unsafe_allow_html=True)

    with c4:
        st.markdown(f"<div class='card orange'><h4>📉 Kategori Rendah</h4><div class='big-number'>{rendah}</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📊 Distribusi Kemampuan Numerasi")

    grafik = df["kategori_numerasi"].value_counts().reset_index()
    grafik.columns = ["Kategori", "Jumlah"]

    fig = px.bar(
        grafik,
        x="Kategori",
        y="Jumlah",
        color="Kategori",
        text="Jumlah",
        color_discrete_map={
            "Rendah": "#F72585",
            "Sedang": "#FFB703",
            "Tinggi": "#2DC653",
            "Belum Diprediksi": "#ADB5BD"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

elif menu == "📝 Input Data Siswa":

    st.title("Input Data Siswa")
    st.caption("Menu ini hanya digunakan untuk memasukkan data siswa baru. Cluster, kategori, dan rekomendasi akan muncul pada menu Prediksi Numerasi.")

    with st.form("form_input"):
        nama = st.text_input("Nama Siswa")
        math = st.number_input("Math Score", min_value=0, max_value=100, value=70)
        reading = st.number_input("Reading Score", min_value=0, max_value=100, value=70)
        writing = st.number_input("Writing Score", min_value=0, max_value=100, value=70)
        submit = st.form_submit_button("Simpan Data")

    if submit:
        if nama.strip() == "":
            st.warning("Nama siswa harus diisi.")
        else:
            siswa_baru = pd.DataFrame({
                "nama siswa": [nama],
                "math score": [math],
                "reading score": [reading],
                "writing score": [writing],
                "cluster": ["Belum Diprediksi"],
                "kategori_numerasi": ["Belum Diprediksi"],
                "rekomendasi": ["Belum Diprediksi"]
            })

            st.session_state.data_baru = pd.concat(
                [st.session_state.data_baru, siswa_baru],
                ignore_index=True
            )

            st.success(f"Data {nama} berhasil disimpan. Silakan lanjut ke menu Prediksi Numerasi.")

    st.subheader("Data Siswa Lama dan Data Baru")
    tampil_input = pd.concat([df_awal, st.session_state.data_baru], ignore_index=True)

    st.dataframe(
        tampil_input[["nama siswa", "math score", "reading score", "writing score"]],
        use_container_width=True
    )

elif menu == "🤖 Prediksi Numerasi":

    st.title("Prediksi Numerasi")
    st.caption("Menu ini menampilkan hasil cluster, kategori numerasi, dan rekomendasi berdasarkan model Random Forest yang mengikuti pola hasil clustering.")

    if len(st.session_state.data_baru) > 0:
        if st.button("Prediksi Semua Data Baru"):
            for i in range(len(st.session_state.data_baru)):
                math = st.session_state.data_baru.loc[i, "math score"]
                reading = st.session_state.data_baru.loc[i, "reading score"]
                writing = st.session_state.data_baru.loc[i, "writing score"]

                data_pred = pd.DataFrame({
                    "math score": [math],
                    "reading score": [reading],
                    "writing score": [writing]
                })

                hasil = model.predict(data_pred)[0]
                rekom = buat_rekomendasi(hasil)

                if hasil == "Rendah":
                    cluster_label = "Cluster Rendah"
                elif hasil == "Sedang":
                    cluster_label = "Cluster Sedang"
                else:
                    cluster_label = "Cluster Tinggi"

                st.session_state.data_baru.loc[i, "cluster"] = cluster_label
                st.session_state.data_baru.loc[i, "kategori_numerasi"] = hasil
                st.session_state.data_baru.loc[i, "rekomendasi"] = rekom

            st.success("Seluruh data baru berhasil diprediksi.")

    df_prediksi = pd.concat([df_awal, st.session_state.data_baru], ignore_index=True)

    st.subheader("Data Hasil Prediksi Numerasi")

    st.dataframe(
        df_prediksi[
            [
                "nama siswa",
                "math score",
                "reading score",
                "writing score",
                "cluster",
                "kategori_numerasi",
                "rekomendasi"
            ]
        ],
        use_container_width=True
    )

elif menu == "📈 Monitoring Perkembangan":

    st.title("Monitoring Perkembangan Siswa")
    st.caption("Menu ini menampilkan monitoring kemampuan numerasi secara keseluruhan dan per siswa.")

    df_monitor = pd.concat([df_awal, st.session_state.data_baru], ignore_index=True)
    df_monitor = df_monitor[df_monitor["kategori_numerasi"] != "Belum Diprediksi"]

    if len(df_monitor) == 0:
        st.warning("Belum ada data yang dapat dimonitoring. Lakukan prediksi terlebih dahulu.")
    else:
        df_monitor["math score"] = pd.to_numeric(df_monitor["math score"], errors="coerce")
        df_monitor["reading score"] = pd.to_numeric(df_monitor["reading score"], errors="coerce")
        df_monitor["writing score"] = pd.to_numeric(df_monitor["writing score"], errors="coerce")

        df_monitor = df_monitor.dropna(
            subset=["math score", "reading score", "writing score"]
        )

        df_monitor["skor numerasi"] = (
            df_monitor["math score"] +
            df_monitor["reading score"] +
            df_monitor["writing score"]
        ) / 3

        pilih = st.selectbox("Pilih Siswa", df_monitor["nama siswa"].astype(str).unique())

        data_siswa = df_monitor[df_monitor["nama siswa"].astype(str) == pilih].iloc[0]

        skor_numerasi = round(float(data_siswa["skor numerasi"]), 1)
        kategori = data_siswa["kategori_numerasi"]

        c1, c2 = st.columns([1, 2])

        with c1:
            st.markdown(f"""
            <div class='white-box'>
                <h3>{pilih}</h3>
                <p>Skor Numerasi</p>
                <h1>{skor_numerasi}</h1>
                <p>Kategori: <b>{kategori}</b></p>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            grafik_siswa = pd.DataFrame({
                "Aspek": ["Math Score", "Reading Score", "Writing Score"],
                "Nilai": [
                    float(data_siswa["math score"]),
                    float(data_siswa["reading score"]),
                    float(data_siswa["writing score"])
                ]
            })

            fig_line = px.line(
                grafik_siswa,
                x="Aspek",
                y="Nilai",
                markers=True,
                title=f"Grafik Nilai Numerasi {pilih}"
            )
            st.plotly_chart(fig_line, use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Visualisasi Keseluruhan Siswa")

        col1, col2 = st.columns(2)

        with col1:
            fig_bar = px.bar(
                df_monitor,
                x="nama siswa",
                y="skor numerasi",
                color="kategori_numerasi",
                title="Skor Numerasi Tiap Siswa",
                color_discrete_map={
                    "Rendah": "#F72585",
                    "Sedang": "#FFB703",
                    "Tinggi": "#2DC653"
                }
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            fig_pie = px.pie(
                df_monitor,
                names="kategori_numerasi",
                title="Persentase Kategori Numerasi",
                color="kategori_numerasi",
                color_discrete_map={
                    "Rendah": "#F72585",
                    "Sedang": "#FFB703",
                    "Tinggi": "#2DC653"
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("📌 Scatter Plot Kemampuan Numerasi")

        df_scatter = df_monitor.copy()

        for col in ["math score", "reading score", "writing score"]:
            df_scatter[col] = pd.to_numeric(df_scatter[col], errors="coerce")

        df_scatter = df_scatter.dropna(
            subset=[
                "math score",
                "reading score",
                "writing score",
                "kategori_numerasi"
            ]
        )

        df_scatter["ukuran_titik"] = df_scatter["writing score"].astype(float).clip(lower=1)

        if len(df_scatter) == 0:
            st.warning("Data scatter plot belum tersedia.")
        else:
            fig_scatter = px.scatter(
                df_scatter,
                x="reading score",
                y="math score",
                size="ukuran_titik",
                color="kategori_numerasi",
                hover_name="nama siswa",
                hover_data={
                    "writing score": True,
                    "ukuran_titik": False
                },
                title="Sebaran Siswa Berdasarkan Reading Score, Math Score, dan Writing Score",
                color_discrete_map={
                    "Rendah": "#F72585",
                    "Sedang": "#FFB703",
                    "Tinggi": "#2DC653"
                }
            )

            fig_scatter.update_layout(height=500)
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("📋 Tabel Monitoring Keseluruhan")

        st.dataframe(
            df_monitor[
                [
                    "nama siswa",
                    "math score",
                    "reading score",
                    "writing score",
                    "skor numerasi",
                    "kategori_numerasi",
                    "rekomendasi"
                ]
            ],
            use_container_width=True
        )

elif menu == "💡 Rekomendasi Pembelajaran":

    st.title("Rekomendasi Pembelajaran")

    df_rekom = pd.concat([df_awal, st.session_state.data_baru], ignore_index=True)
    df_rekom = df_rekom[df_rekom["kategori_numerasi"] != "Belum Diprediksi"]

    if len(df_rekom) == 0:
        st.warning("Belum ada rekomendasi. Lakukan prediksi terlebih dahulu.")
    else:
        df_rekom["math score"] = pd.to_numeric(df_rekom["math score"], errors="coerce")
        df_rekom["reading score"] = pd.to_numeric(df_rekom["reading score"], errors="coerce")
        df_rekom["writing score"] = pd.to_numeric(df_rekom["writing score"], errors="coerce")

        df_rekom = df_rekom.dropna(
            subset=["math score", "reading score", "writing score"]
        )

        df_rekom["skor numerasi"] = (
            df_rekom["math score"] +
            df_rekom["reading score"] +
            df_rekom["writing score"]
        ) / 3

        pilih_rekom = st.selectbox("Pilih Siswa", df_rekom["nama siswa"].astype(str).unique())

        data_rekom = df_rekom[df_rekom["nama siswa"].astype(str) == pilih_rekom].iloc[0]

        skor = round(float(data_rekom["skor numerasi"]), 1)
        kategori = data_rekom["kategori_numerasi"]
        rekom = data_rekom["rekomendasi"]

        if kategori == "Rendah":
            badge_class = "badge-rendah"
        elif kategori == "Sedang":
            badge_class = "badge-sedang"
        else:
            badge_class = "badge-tinggi"

        st.markdown(f"""
        <div class='rekom-card'>
            <h3>{pilih_rekom}</h3>
            <p>Skor Numerasi: <b>{skor}</b></p>
            <span class='badge {badge_class}'>{kategori}</span>
            <br><br>
            <p><b>Tindak lanjut:</b> {rekom}</p>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Daftar Rekomendasi Semua Siswa")
        st.dataframe(
            df_rekom[
                [
                    "nama siswa",
                    "skor numerasi",
                    "kategori_numerasi",
                    "rekomendasi"
                ]
            ],
            use_container_width=True
        )