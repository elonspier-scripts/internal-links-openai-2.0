import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import re
import io
import json
from urllib.parse import urlparse

# ========================================================
# 1. UI CONFIGURATIE (Deep Black / Cyber Blue)
# ========================================================
st.set_page_config(page_title="SEO Link Matrix Pro", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #050505 !important; border-right: 1px solid #1e1e1e; }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #0a0a0a !important; color: #00a2ff !important; border: 1px solid #222 !important;
    }
    .stDataFrame, div[data-testid="stTable"] { 
        background-color: #000 !important; border: 1px solid #1e1e1e !important; border-radius: 4px;
    }
    h1, h2, h3 { color: #00a2ff !important; font-family: 'Inter', sans-serif; }
    .stButton>button { 
        background: linear-gradient(135deg, #0044ff 0%, #00a2ff 100%);
        color: white; border: none; padding: 12px; font-weight: bold; width: 100%;
        box-shadow: 0 4px 20px rgba(0, 162, 255, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# ========================================================
# 2. INITIALISATIE
# ========================================================
if 'df_results' not in st.session_state:
    st.session_state.df_results = None

with st.sidebar:
    st.title("⚙️ Configuratie")
    api_key = st.text_input("OpenAI API Key", type="password", key="api_key_val")
    st.divider()
    
    cluster_threshold = st.slider("Minimale Cluster Match % (Hubs)", 50, 95, 80) / 100
    score_threshold = st.slider("Minimale Link Match % (Links)", 50, 95, 80) / 100
    links_per_page = st.slider("Aantal links per URL", 1, 10, 5)

# ========================================================
# 3. HELPERS (Met Caching voor Performance)
# ========================================================
@st.cache_data(show_spinner=False)
def clean_path(url):
    path = url.split('/')[-1] if not url.strip().endswith('/') else url.split('/')[-2]
    return re.sub(r'[-_/]', ' ', path)

@st.cache_data(show_spinner=False)
def get_folder(url):
    """Extraheert de eerste hoofdmap (top-level folder) uit een URL"""
    try:
        path = urlparse(str(url)).path
        clean_path = path.strip('/')
        if not clean_path:
            return '/'
        first_folder = clean_path.split('/')[0]
        return f"/{first_folder}/"
    except:
        return "/"

def get_embeddings(texts, key, batch_size=500):
    """Haalt embeddings op in batches voor snelheid en stabiliteit"""
    client = OpenAI(api_key=key)
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        res = client.embeddings.create(input=batch, model='text-embedding-3-small')
        all_embeddings.extend([d.embedding for d in res.data])
    return np.array(all_embeddings)

def get_ai_cluster_names_bulk(clusters_dict, key):
    client = OpenAI(api_key=key)
    payload = ""
    for cid, texts in clusters_dict.items():
        sample = "\n".join(texts[:10])
        payload += f"Cluster ID {cid}:\n{sample}\n\n"
        
    prompt = f"""
    Je bent een SEO expert. Hieronder staan verschillende clusters met URL's en titels.
    Bedenk voor ELK cluster één overkoepelende categorie-naam.
    Regels: Max 5 woorden, korte beschrijvende naam, ALTIJD in JSON formaat met Cluster ID als key.
    Hier is de data:
    {payload}
    """
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={ "type": "json_object" }
    )
    try:
        result = json.loads(res.choices[0].message.content)
        return {int(k): str(v) for k, v in result.items()}
    except:
        return {}

def color_score(v):
    if not isinstance(v, (int, float)): return ''
    if v >= 85: return 'color: #28a745; font-weight: bold;'
    elif v >= 70: return 'color: #ffc107; font-weight: bold;'
    else: return 'color: #dc3545; font-weight: bold;'

# ========================================================
# 4. DASHBOARD TABS
# ========================================================
st.title("🔗 SEO Link Intelligence Matrix")
tab_tool, tab_inst = st.tabs(["🚀 Analyse Tool", "📖 Instructies"])

with tab_inst:
    st.header("Hoe gebruik je deze tool?")
    st.markdown("1. Upload Website CSV. 2. Voer OpenAI Key in. 3. Plak Focus URL's.")

with tab_tool:
    c1, c2 = st.columns([1, 1])
    with c1:
        file = st.file_uploader("1. Upload Website CSV", type=['csv'], key="csv_uploader")
    with c2:
        urls_txt = st.text_area("2. Focus URL's (één per regel)", key="urls_input", height=100)

    # ========================================================
    # 5. DE ANALYSE ENGINE (Snelheids-Geoptimaliseerd)
    # ========================================================
    if st.button("🚀 GENEREER INTELLIGENCE MATRIX"):
        if not api_key or not file or not urls_txt:
            st.error("⚠️ Vul alle velden in (API Key, CSV en Focus URL's).")
        else:
            try:
                with st.spinner("Bezig met AI-analyse en clustering..."):
                    raw_df = pd.read_csv(file)
                    url_col = raw_df.columns[0]
                    focus_list = [u.strip() for u in urls_txt.split('\n') if u.strip()]
                    
                    # Opschonen data
                    clean_df = raw_df.dropna(subset=[url_col]).copy()
                    clean_df = clean_df.fillna("")
                    clean_df = clean_df[clean_df[url_col].astype(str).str.strip() != ""]
                    
                    # Tekst voorbereiden
                    clean_df['text'] = clean_df[url_col].astype(str).apply(clean_path) + " " + clean_df.iloc[:, 1].astype(str)
                    
                    # 1. Embeddings (Batch verwerking)
                    vecs = get_embeddings(clean_df['text'].tolist(), api_key)
                    
                    # 2. Clustering
                    dist_thresh = 1.0 - cluster_threshold
                    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=dist_thresh, metric='cosine', linkage='average')
                    clean_df['Cluster_ID'] = clustering_model.fit_predict(vecs)
                    
                    # 3. AI Cluster Naming (Bulk)
                    clusters_to_name = {cid: clean_df[clean_df['Cluster_ID'] == cid]['text'].tolist() for cid in clean_df['Cluster_ID'].unique()}
                    cluster_names = get_ai_cluster_names_bulk(clusters_to_name, api_key)
                    
                    clean_df['Category'] = clean_df['Cluster_ID'].apply(lambda x: cluster_names.get(x, "ALGEMEEN"))
                    cat_lookup = dict(zip(clean_df[url_col], clean_df['Category']))

                    # 4. Interne Links (Numpy Versnelling)
                    sims = cosine_similarity(vecs)
                    urls_array = clean_df[url_col].values # Snelheids-upgrade

                    found = []
                    for f_url in focus_list:
                        if f_url not in clean_df[url_col].values: continue
                        idx_src = clean_df.index[clean_df[url_col] == f_url].tolist()[0]
                        src_cat = clean_df.iloc[idx_src]['Category']
                        
                        scores = sims[idx_src]
                        top_idx = np.argsort(scores)[::-1]
                        
                        added = 0
                        for t_idx in top_idx:
                            s = float(scores[t_idx])
                            if s < score_threshold: break # Snelheids-upgrade: stop als score te laag wordt
                                
                            t_url = urls_array[t_idx]
                            if f_url != t_url:
                                found.append({
                                    'From Hub': src_cat,
                                    'From Folder': get_folder(f_url),
                                    'Focus URL': f_url,
                                    'To Hub': cat_lookup.get(t_url, "ALGEMEEN"),
                                    'To Folder': get_folder(t_url),
                                    'Target URL': t_url,
                                    'Score': s * 100
                                })
                                added += 1
                                if added >= links_per_page: break

                    st.session_state.df_results = pd.DataFrame(found)
                    st.rerun()

            except Exception as e:
                st.error(f"Systeemfout: {e}")
                
    # ========================================================
    # 6. INTERACTIEVE MATRIX & OUTPUT
    # ========================================================
    if st.session_state.df_results is not None:
        data = st.session_state.df_results
        st.divider()
        st.subheader("📊 Cross-Linking Matrices (Intensity)")
        
        tab_matrix_hub, tab_matrix_folder = st.tabs(["🗂️ Semantische Hub Matrix", "📁 Technische Folder Matrix"])

        def style_matrix_cells(val, mx_val):
            if val == 0: return 'background-color: #0a0a0a; color: #222222; text-align: center;'
            intensity = 0.2 + 0.8 * (val / mx_val)
            return f'background-color: rgba(0, 162, 255, {intensity}); color: #ffffff; font-weight: bold; text-align: center;'

        # HUB MATRIX
        with tab_matrix_hub:
            matrix_hub = pd.crosstab(data['From Hub'], data['To Hub'])
            matrix_hub = matrix_hub.reindex(index=matrix_hub.sum(axis=1).sort_values(ascending=False).index, 
                                            columns=matrix_hub.sum(axis=0).sort_values(ascending=False).index, fill_value=0)
            st.dataframe(matrix_hub.style.map(lambda v: style_matrix_cells(v, matrix_hub.values.max() if matrix_hub.values.max()>0 else 1)), 
                         width='content', on_select="rerun", selection_mode="single-row", key="mx_hub")

            sel_hub = st.session_state.get("mx_hub")
            if sel_hub and sel_hub.get("selection", {}).get("rows"):
                f_cat = matrix_hub.index[sel_hub["selection"]["rows"][0]]
                st.markdown(f"### 🎯 Hub: `{f_cat}`")
                df_f = data[data['From Hub'] == f_cat][['Focus URL', 'To Hub', 'Target URL', 'Score']].sort_values(by='Score', ascending=False)
                st.dataframe(df_f.style.map(color_score, subset=['Score']), width='content', hide_index=True)

        # FOLDER MATRIX
        with tab_matrix_folder:
            matrix_folder = pd.crosstab(data['From Folder'], data['To Folder'])
            matrix_folder = matrix_folder.reindex(index=matrix_folder.sum(axis=1).sort_values(ascending=False).index, 
                                               columns=matrix_folder.sum(axis=0).sort_values(ascending=False).index, fill_value=0)
            st.dataframe(matrix_folder.style.map(lambda v: style_matrix_cells(v, matrix_folder.values.max() if matrix_folder.values.max()>0 else 1)), 
                         width='content', on_select="rerun", selection_mode="single-row", key="mx_folder")

            sel_folder = st.session_state.get("mx_folder")
            if sel_folder and sel_folder.get("selection", {}).get("rows"):
                f_folder = matrix_folder.index[sel_folder["selection"]["rows"][0]]
                st.markdown(f"### 🎯 Folder: `{f_folder}`")
                df_ff = data[data['From Folder'] == f_folder][['Focus URL', 'To Folder', 'Target URL', 'Score']].sort_values(by='Score', ascending=False)
                st.dataframe(df_ff.style.map(color_score, subset=['Score']), width='content', hide_index=True)

        # 7. OVERZICHT & EXPORT
        st.divider()
        st.subheader("🏗️ Topic Hubs Overzicht")
        hub_stats = data.groupby('From Hub')['Score'].mean().sort_values(ascending=False)
        t1, t2, t3 = st.tabs(["🟢 Sterk (>= 85%)", "🟡 Gemiddeld (70-84%)", "🔴 Zwak (< 70%)"])
        
        def show_hubs(stats_series):
            for hub, avg in stats_series.items():
                with st.expander(f"📁 {hub} ({round(avg)}%)"):
                    st.dataframe(data[data['From Hub'] == hub][['Focus URL', 'Target URL', 'Score']].style.map(color_score, subset=['Score']), width='content', hide_index=True)

        with t1: show_hubs(hub_stats[hub_stats >= 85])
        with t2: show_hubs(hub_stats[(hub_stats >= 70) & (hub_stats < 85)])
        with t3: show_hubs(hub_stats[hub_stats < 70])

        st.divider()
        csv_buf = io.StringIO()
        data.to_csv(csv_buf, index=False, sep=';')
        st.download_button("📥 Download CSV", csv_buf.getvalue(), "seo_links.csv", "text/csv")
