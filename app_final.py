import streamlit as st
import torch
import os
import sys
import pickle
import gdown
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- 0. Page Configuration (MUST BE FIRST) ---
st.set_page_config(page_title="🇮🇩 Detektor Komentar Negatif", layout="wide")

# --- 1. Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')

if src_dir not in sys.path:
    sys.path.append(src_dir)

try:
    from model_pipeline import TwoStageModelPipeline 
    from data_preprocessing import DataPreprocessor
except ImportError as e:
    st.error(f"Import Error: Could not load dependencies. Original Error: {e}")
    st.stop()

# --- 2. Global Configurations ---
MODEL_DIR = os.path.join(current_dir, 'models')
RF_MODEL_PATH = os.path.join(MODEL_DIR, 'rf_model.pkl')
TFIDF_VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_sastrawi.pkl')
BERT_WEIGHTS_PATH = os.path.join(MODEL_DIR, 'bert_model_cpu.pkl')

BERT_BASE_NAME = "indobenchmark/indobert-base-p2" 
TFIDF_GDRIVE_ID = "1j1XQuEPRslvQHC6Mq1S-ROPLrrkwAKRV"
RF_MODEL_GDRIVE_ID = "1hPgOxAC6U7Pgu9A5FoJnaiYavGh3Z0Jt"
BERT_MODEL_GDRIVE_ID = "1BOBydauMfn5eO087Eo4WUbkCT7tBa9uM"

LABEL_MAP = {
    0: "Provokasi", 1: "Penghinaan", 2: "Pornografi",
    3: "Negatif Lainnya", 4: "SARA", 5: "Pencemaran Nama Baik"
}
LABEL_NAME_MAP = {v: k for v, k in LABEL_MAP.items()}

WARNING_MSG = {
    "Pornografi" : "Konten terdeteksi mengarah ke **pornografi**. Penyebaran konten asusila melanggar Pasal 27 ayat (1) UU ITE dan UU Pornografi. Hapus atau ubah pesan ini segera.",
    "Pencemaran Nama Baik" : "Pesan ini terindikasi **pencemaran nama baik**. Perbuatan ini diancam pidana berdasarkan Pasal 27 ayat (3) UU ITE atau Pasal 310 KUHP. Mohon ubah pernyataan Anda.",
    "Provokasi" : "Pesan ini terindikasi **provokasi/penghasutan**. Mengajak melakukan tindak pidana diatur dalam Pasal 160 KUHP. Hati-hati, ini termasuk tindak kriminal.",
    "SARA" : "Pesan ini terindikasi **ujaran kebencian (SARA)**. Menyebarkan informasi SARA melanggar Pasal 28 ayat (2) UU ITE. Segera ubah pesan Anda.",
    "Penghinaan" : "Pesan ini terindikasi **penghinaan**. Penghinaan tetap memiliki konsekuensi hukum. Pasal 315 KUHP berlaku.",
    "Negatif Lainnya" : "Pesan ini mengandung **konten negatif**. Walaupun belum terklasifikasi spesifik, pesan ini berpotensi melanggar peraturan pidana. Harap bijak berkomentar!"
}

def download_model(file_path: str, gdrive_id: str):
    if os.path.exists(file_path):
        return
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    url = f"https://drive.google.com/uc?id={gdrive_id}"
    with st.spinner(f"Downloading {os.path.basename(file_path)}..."):
        gdown.download(url, file_path, quiet=False)

# --- 3. Model Loading (Cached) ---
@st.cache_resource
def load_full_pipeline():
    download_model(TFIDF_VECTORIZER_PATH, TFIDF_GDRIVE_ID)
    download_model(RF_MODEL_PATH, RF_MODEL_GDRIVE_ID)
    download_model(BERT_WEIGHTS_PATH, BERT_MODEL_GDRIVE_ID)
    
    try:
        pipeline = TwoStageModelPipeline(
            rf_model_path=RF_MODEL_PATH,
            tfidf_path=TFIDF_VECTORIZER_PATH,
            bert_model_path=BERT_BASE_NAME, 
            label_map=LABEL_NAME_MAP 
        )
        
        state_dict = torch.load(BERT_WEIGHTS_PATH, map_location=torch.device("cpu"))
        pipeline.bert_model.load_state_dict(state_dict)
        pipeline.bert_model.eval()
        return pipeline
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

pipeline = load_full_pipeline()

# --- 4. Logic Functions ---
if 'final_result' not in st.session_state:
    st.session_state.final_result = None

def classify_text():
    text = st.session_state.input_widget
    if text:
        with st.spinner('Sedang memproses...'):
            preprocessor = DataPreprocessor()
            processed_text = preprocessor.full_process(text)
            final_label = pipeline.predict(processed_text) 
            st.session_state.final_result = final_label
    else:
        st.session_state.final_result = None

# --- 5. UI Helper ---
def draw_detector_ui(result_label=None, sub_label=None):
    st.markdown("""
        <style>
            .result-neg-main { background-color: #B50000; color: white; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; font-size: 1.2em; }
            .result-neg-sub { background-color: #fce7e7; color: #d9363e; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; font-size: 1.2em; border: 1px solid #d9363e;}
            .result-non-neg { background-color: #8f8; color: black; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; font-size: 1.2em; }
        </style>
    """, unsafe_allow_html=True)
    
    if result_label:
        st.subheader("HASIL ANALISIS")
        st.write("Komentar Anda tergolong:")

        if result_label == "NEGATIF":
            st.markdown(f'<div class="result-neg-main">{result_label}</div>', unsafe_allow_html=True)
            if sub_label:
                st.write("")
                st.write("Terklasifikasi dalam subkategori:")
                st.markdown(f'<div class="result-neg-sub">{sub_label.upper()}</div>', unsafe_allow_html=True)
                st.write("")
                if sub_label in WARNING_MSG:
                    st.error(WARNING_MSG[sub_label])
                else:
                    st.info("Pesan ini berpotensi melanggar peraturan pidana. Harap bijak berkomentar!")
            st.write("⚠️ **Mohon ubah komentar Anda sebelum di-post ke media sosial!**")
            
        elif result_label == "NON-NEGATIF":
            st.markdown(f'<div class="result-non-neg">{result_label}</div>', unsafe_allow_html=True)
            st.write("")
            st.write("Aman untuk di-post ke media sosial 👍")

# --- 6. Sidebar ---
with st.sidebar:
    st.header("⚖️ Penjelasan Subkategori")
    st.write("Klik untuk melihat dasar hukum.")
    with st.expander("Pornografi"):
        st.markdown("**Definisi:** Pesan bermuatan kecabulan.\n\n**Hukum:** UU Pornografi No. 44/2008 & Pasal 27(1) UU ITE.")
    with st.expander("Pencemaran Nama Baik"):
        st.markdown("**Definisi:** Merusak reputasi orang lain.\n\n**Hukum:** Pasal 310 KUHP & Pasal 27(3) UU ITE.")
    with st.expander("Provokasi"):
        st.markdown("**Definisi:** Menghasut tindak pidana.\n\n**Hukum:** Pasal 160 KUHP.")
    with st.expander("SARA"):
        st.markdown("**Definisi:** Kebencian berbasis Suku/Agama/Ras.\n\n**Hukum:** Pasal 28(2) UU ITE.")
    with st.expander("Penghinaan"):
        st.markdown("**Definisi:** Merendahkan martabat.\n\n**Hukum:** Pasal 315 KUHP.")

# --- 7. Main App Layout ---
st.title("🇮🇩 DETEKTOR KOMENTAR NEGATIF")
st.markdown("Sistem ini mendeteksi ujaran kebencian dan konten negatif dalam Bahasa Indonesia.")
st.info("💡 **Pro tip**: Buka sidebar untuk melihat penjelasan hukum tiap kategori.")

# Input
st.subheader("Masukkan Teks Komentar")
st.text_area(
    "Teks Komentar:",
    placeholder="Masukkan kalimat atau emoji di sini...",
    key='input_widget',
    height=120,
    label_visibility="collapsed"
)

st.markdown("""
    **Contoh Input:**
    1. Emoji: 😠🤬 | 2. Slang: anj**, b**g**t | 3. Kalimat: Muka lu kayak babi
""")

# Button
st.button("Proses Kalimat", type="primary", on_click=classify_text)

st.markdown("---")

# Result Display
if st.session_state.final_result:
    res = st.session_state.final_result
    if res == "Non-Negatif":
        draw_detector_ui(result_label='NON-NEGATIF')
    else:
        draw_detector_ui(result_label='NEGATIF', sub_label=res)