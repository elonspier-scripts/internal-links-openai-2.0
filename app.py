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
# 3. HELPERS
# ========================================================
def clean_path(url):
    path = url.split('/')[-1] if not url.strip().endswith('/') else url.split('/')[-2]
    return re.sub(r'[-_/]', ' ', path)

def get_folder(url):
    """Extraheert de eerste hoofdmap (top-level folder) uit een URL"""
    try:
        path = urlparse(str(url)).path
        clean_path = path.strip('/')
        
        # Als er na het strippen niks overblijft, is het de root
        if not clean_path:
            return '/'
            
        # Pak altijd de eerste submap, ongeacht hoe diep de URL gaat
        first_folder = clean_path.split('/')[0]
        return f"/{first_folder}/"
    except:
        return "/"

def get_embeddings(texts, key):
    client = OpenAI(api_key=key)
    res = client.embeddings.create(input=texts, model='text-embedding-3-small')
    return np.array([d.embedding for d in res.data])

def get_ai_cluster_names_bulk(clusters_dict, key):
    client = OpenAI(api_key=key)
    
    payload = ""
    for cid, texts in clusters_dict.items():
        sample = "\n".join(texts[:10])
        payload += f"Cluster ID {cid}:\n{sample}\n\n"
        
    prompt = f"""
    Je bent een SEO expert. Hieronder staan verschillende clusters met URL's en titels.
    Bedenk voor ELK cluster één overkoepelende categorie-naam.
    
    Regels:
    - Maximaal 5 woorden per naam.
    - Geef het een korte beschrijvende / relevante naam
    - Geef je antwoord ALTIJD terug in strict JSON formaat. De sleutel (key) is het Cluster ID (bijv. "0", "1") en de waarde (value) is de bedachte naam.
    
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
    except Exception as e:
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
    st.markdown("""
    ### 1. Voorbereiding van het CSV-bestand
    Lever een CSV-bestand aan met de volgende structuur:
    * **Kolom A (eerste kolom):** Moet de volledige URL's bevatten.
    * **Overige kolommen:** Hier mag content staan zoals de Page Title, H1 of de hoofdtekst.
    
    ### 2. OpenAI API Key
    Voer je eigen OpenAI API Key in de sidebar in.
    
    ### 3. Focus URL's
    Plak in het tekstveld de URL's die je wilt analyseren. Gebruik één URL per regel.
    """)

with tab_tool:
    c1, c2 = st.columns([1, 1])
    with c1:
        file = st.file_uploader("1. Upload Website CSV", type=['csv'], key="csv_uploader")
    with c2:
        urls_txt = st.text_area("2. Focus URL's (één per regel)", key="urls_input", height=100)

    # ========================================================
    # 5. DE ANALYSE ENGINE
    # ========================================================
    if st.button("🚀 GENEREER INTELLIGENCE MATRIX"):
        missing = []
        if not api_key: missing.append("OpenAI API Key (in de sidebar)")
        if not file: missing.append("CSV-bestand")
        if not urls_txt: missing.append("Focus URL's")

        if missing:
            st.error(f"⚠️ De volgende velden ontbreken: {', '.join(missing)}")
        else:
            try:
                with st.spinner("Bezig met AI-clustering en semantische analyse... Dit kan even duren."):
                    raw_df = pd.read_csv(file)
                    url_col = raw_df.columns[0]
                    focus_list = [u.strip() for u in urls_txt.split('\n') if u.strip()]
                    
                    clean_df = raw_df.dropna(subset=[url_col]).copy()
                    clean_df = clean_df.fillna("")
                    clean_df = clean_df[clean_df[url_col].astype(str).str.strip() != ""]
                    
                    clean_df['text'] = clean_df[url_col].astype(str).apply(clean_path) + " " + clean_df.iloc[:, 1].astype(str)
                    
                    vecs = get_embeddings(clean_df['text'].tolist(), api_key)
                    
                    dist_thresh = 1.0 - cluster_threshold
                    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=dist_thresh, metric='cosine', linkage='average')
                    clean_df['Cluster_ID'] = clustering_model.fit_predict(vecs)
                    
                    clusters_to_name = {}
                    for cluster_id in clean_df['Cluster_ID'].unique():
                        texts_in_cluster = clean_df[clean_df['Cluster_ID'] == cluster_id]['text'].tolist()
                        clusters_to_name[cluster_id] = texts_in_cluster
                    
                    cluster_names = get_ai_cluster_names_bulk(clusters_to_name, api_key)
                    
                    clean_df['Category'] = clean_df['Cluster_ID'].apply(lambda x: cluster_names.get(x, "ALGEMEEN"))
                    cat_lookup = dict(zip(clean_df[url_col], clean_df['Category']))

                    sims = cosine_similarity(vecs)

                    found = []
                    for f_url in focus_list:
                        if f_url not in clean_df[url_col].values: continue
                        idx_src = clean_df.index[clean_df[url_col] == f_url].tolist()[0]
                        src_cat = clean_df.iloc[idx_src]['Category']
                        
                        scores = sims[idx_src]
                        top_idx = np.argsort(scores)[::-1]
                        
                        added = 0
                        for t_idx in top_idx:
                            t_url = clean_df.iloc[t_idx][url_col]
                            s = float(scores[t_idx])
                            if f_url != t_url and s >= score_threshold:
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
        st.info("💡 Klik op een rij om de details te zien. De matrix is gesorteerd op volume.")
        
        # Maak twee tabbladen aan voor de Matrix weergave
        tab_matrix_hub, tab_matrix_folder = st.tabs(["🗂️ Semantische Hub Matrix", "📁 Technische Folder Matrix"])

        def style_matrix_cells(val, mx_val):
            if val == 0:
                return 'background-color: #0a0a0a; color: #222222; text-align: center;'
            else:
                intensity = 0.2 + 0.8 * (val / mx_val)
                return f'background-color: rgba(0, 162, 255, {intensity}); color: #ffffff; font-weight: bold; text-align: center;'

        # --- TAB 1: HUB MATRIX ---
        with tab_matrix_hub:
            matrix_hub = pd.crosstab(data['From Hub'], data['To Hub'])
            row_order_hub = matrix_hub.sum(axis=1).sort_values(ascending=False).index
            col_order_hub = matrix_hub.sum(axis=0).sort_values(ascending=False).index
            matrix_hub = matrix_hub.reindex(index=row_order_hub, columns=col_order_hub, fill_value=0)

            max_val_hub = matrix_hub.values.max() if matrix_hub.values.max() > 0 else 1
            styled_matrix_hub = matrix_hub.style.map(lambda v: style_matrix_cells(v, max_val_hub))

            st.dataframe(
                styled_matrix_hub,
                width='content',
                on_select="rerun",
                selection_mode="single-row",
                key="matrix_selector_hub"
            )

            selection_hub = st.session_state.get("matrix_selector_hub")
            if selection_hub and selection_hub.get("selection", {}).get("rows"):
                selected_idx = selection_hub["selection"]["rows"][0]
                f_cat = matrix_hub.index[selected_idx]
                
                st.markdown(f"### 🎯 Uitgaande links vanuit Hub: `{f_cat}`")
                filtered = data[data['From Hub'] == f_cat]
                display_filtered = filtered[['Focus URL', 'To Hub', 'Target URL', 'Score']].sort_values(by=['Focus URL', 'Score'], ascending=[True, False]).copy()
                display_filtered.loc[display_filtered.duplicated('Focus URL'), 'Focus URL'] = ""
                
                st.dataframe(
                    display_filtered.style.map(color_score, subset=['Score']),
                    width='content',
                    hide_index=True,
                    column_config={"Score": st.column_config.NumberColumn(format="%d%%")}
                )

        # --- TAB 2: FOLDER MATRIX ---
        with tab_matrix_folder:
            matrix_folder = pd.crosstab(data['From Folder'], data['To Folder'])
            row_order_folder = matrix_folder.sum(axis=1).sort_values(ascending=False).index
            col_order_folder = matrix_folder.sum(axis=0).sort_values(ascending=False).index
            matrix_folder = matrix_folder.reindex(index=row_order_folder, columns=col_order_folder, fill_value=0)

            max_val_folder = matrix_folder.values.max() if matrix_folder.values.max() > 0 else 1
            styled_matrix_folder = matrix_folder.style.map(lambda v: style_matrix_cells(v, max_val_folder))

            st.dataframe(
                styled_matrix_folder,
                width='content',
                on_select="rerun",
                selection_mode="single-row",
                key="matrix_selector_folder"
            )

            selection_folder = st.session_state.get("matrix_selector_folder")
            if selection_folder and selection_folder.get("selection", {}).get("rows"):
                selected_idx = selection_folder["selection"]["rows"][0]
                f_folder = matrix_folder.index[selected_idx]
                
                st.markdown(f"### 🎯 Uitgaande links vanuit Folder: `{f_folder}`")
                filtered_folder = data[data['From Folder'] == f_folder]
                display_filtered_folder = filtered_folder[['Focus URL', 'To Folder', 'Target URL', 'Score']].sort_values(by=['Focus URL', 'Score'], ascending=[True, False]).copy()
                display_filtered_folder.loc[display_filtered_folder.duplicated('Focus URL'), 'Focus URL'] = ""
                
                st.dataframe(
                    display_filtered_folder.style.map(color_score, subset=['Score']),
                    width='content',
                    hide_index=True,
                    column_config={"Score": st.column_config.NumberColumn(format="%d%%")}
                )

        # ========================================================
        # 7. TOPIC HUBS OVERZICHT (Indeling op Sterkte)
        # ========================================================
        st.divider()
        st.subheader("🏗️ Topic Hubs Overzicht")

        hub_stats = data.groupby('From Hub')['Score'].mean().sort_values(ascending=False)

        tab_strong, tab_avg, tab_weak = st.tabs([
            "🟢 Sterk (>= 85%)", 
            "🟡 Gemiddeld (70-84%)", 
            "🔴 Zwak (< 70%)"
        ])

        def render_hub_group(hubs_series):
            if hubs_series.empty:
                st.info("Geen hubs gevonden voor deze categorie.")
            else:
                for hub, avg_score in hubs_series.items():
                    hub_df = data[data['From Hub'] == hub]
                    with st.expander(f"📁 HUB: {hub} ({round(avg_score)}%)"):
                        display_hub = hub_df[['Focus URL', 'To Hub', 'Target URL', 'Score']].sort_values(by=['Focus URL', 'Score'], ascending=[True, False]).copy()
                        display_hub.loc[display_hub.duplicated('Focus URL'), 'Focus URL'] = ""
                        
                        st.dataframe(
                            display_hub.style.map(color_score, subset=['Score']),
                            width='content',
                            hide_index=True,
                            column_config={"Score": st.column_config.NumberColumn(format="%d%%")}
                        )

        with tab_strong:
            strong_hubs = hub_stats[hub_stats >= 85]
            render_hub_group(strong_hubs)

        with tab_avg:
            avg_hubs = hub_stats[(hub_stats >= 70) & (hub_stats < 85)]
            render_hub_group(avg_hubs)

        with tab_weak:
            weak_hubs = hub_stats[hub_stats < 70]
            render_hub_group(weak_hubs)
        
        # ========================================================
        # 8. EXPORT CSV
        # ========================================================
        st.divider()
        export_df = data.copy()
        export_df = export_df.sort_values(by=['From Hub', 'Focus URL', 'Score'], ascending=[True, True, False])
        export_df['Score'] = export_df['Score'].apply(lambda x: f"{round(x)}%")
        
        export_df.loc[export_df.duplicated(subset=['From Hub', 'Focus URL']), 'Focus URL'] = ""
        export_df.loc[export_df.duplicated(subset=['From Hub']), 'From Hub'] = ""
        
        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False, sep=';')
        
        st.download_button(
            label="📥 Download Resultaten (CSV)",
            data=csv_buffer.getvalue(),
            file_name="seo_internal_links_matrix.csv",
            mime="text/csv"
        )
