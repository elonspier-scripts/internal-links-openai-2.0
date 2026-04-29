import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import re
import io
import json

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
    score_threshold = st.slider("Minimale Match %", 50, 95, 80) / 100
    links_per_page = st.slider("Aantal links per URL", 1, 10, 5)

# ========================================================
# 3. HELPERS
# ========================================================
def clean_path(url):
    path = url.split('/')[-1] if not url.strip().endswith('/') else url.split('/')[-2]
    return re.sub(r'[-_/]', ' ', path)

def get_embeddings(texts, key):
    client = OpenAI(api_key=key)
    res = client.embeddings.create(input=texts, model='text-embedding-3-small')
    return np.array([d.embedding for d in res.data])

def get_ai_cluster_names_bulk(clusters_dict, key):
    client = OpenAI(api_key=key)
    
    # We bouwen één groot tekstblok met een paar voorbeelden uit elk cluster
    payload = ""
    for cid, texts in clusters_dict.items():
        sample = "\n".join(texts[:10])  # Max 10 URL's per cluster om tokens te besparen
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
    
    # Gebruik response_format JSON zodat we altijd een leesbaar data-object terugkrijgen
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={ "type": "json_object" }
    )
    
    try:
        # Zet de JSON string van OpenAI om naar een Python dictionary
        result = json.loads(res.choices[0].message.content)
        # OpenAI geeft keys vaak als string terug ("0"), we zetten dit terug naar integers (0)
        return {int(k): str(v) for k, v in result.items()}
    except Exception as e:
        return {} # Fallback als er iets fout gaat

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
    Lever een CSV-bestand aan (bijv. een export uit Screaming Frog of een eigen lijst) met de volgende structuur:
    * **Kolom A (eerste kolom):** Moet de volledige URL's bevatten.
    * **Overige kolommen:** Hier mag content staan zoals de Page Title, H1 of de hoofdtekst. De tool gebruikt deze data om de context te begrijpen.
    
    ### 2. OpenAI API Key
    Voer je eigen OpenAI API Key in de sidebar aan de linkerkant in. De tool maakt gebruik van de `text-embedding-3-small` engine voor razendsnelle en goedkope analyses.

    ### 3. Focus URL's
    Plak in het tekstveld de URL's die je wilt analyseren. Dit zijn de pagina's waarvoor je interne linkmogelijkheden wilt vinden. Gebruik één URL per regel.

    ### 4. De Matrix gebruiken
    * Na de analyse verschijnt een **Cross-Linking Matrix**. 
    * De matrix is standaard gesorteerd op relevantie (de hubs met de meeste kansen staan bovenaan).
    * Klik op een **rij** in de matrix om direct alle specifieke link-kansen voor die hub te openen.
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
                    
                    # --- FIX VOOR LEGE CELLEN (NaN) ---
                    # 1. Verwijder rijen waar de URL compleet leeg is
                    clean_df = raw_df.dropna(subset=[url_col]).copy()
                    
                    # 2. Vul alle overige lege cellen met een lege string (voorkomt 'NaN')
                    clean_df = clean_df.fillna("")
                    
                    # 3. Filter voor de zekerheid URLs weg die alleen uit spaties bestaan
                    clean_df = clean_df[clean_df[url_col].astype(str).str.strip() != ""]
                    # ----------------------------------
                    
                    clean_df['text'] = clean_df[url_col].astype(str).apply(clean_path) + " " + clean_df.iloc[:, 1].astype(str)
                    
                    # 1. Embeddings ophalen voor álle pagina's
                    vecs = get_embeddings(clean_df['text'].tolist(), api_key)
                    
                    # 2. DYNAMISCHE CLUSTERING (AI groepeert de pagina's zelf op basis van Cosine Similarity)
                    # distance_threshold bepaalt hoe streng hij is. 0.5 betekent: redelijk wat vrijheid voor de groottes van de hubs.
                    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5, metric='cosine', linkage='average')
                    clean_df['Cluster_ID'] = clustering_model.fit_predict(vecs)
                    
                    # 3. CLUSTERS EEN NAAM GEVEN (In Bulk!)
                    # Verzamel eerst alle teksten per cluster
                    clusters_to_name = {}
                    for cluster_id in clean_df['Cluster_ID'].unique():
                        texts_in_cluster = clean_df[clean_df['Cluster_ID'] == cluster_id]['text'].tolist()
                        clusters_to_name[cluster_id] = texts_in_cluster
                    
                    # Roep de AI slechts ÉÉN KEER aan voor alle clusters tegelijk
                    cluster_names = get_ai_cluster_names_bulk(clusters_to_name, api_key)
                    
                    # Koppel de AI-namen aan de DataFrame. Als een naam ontbreekt, gebruik "ALGEMEEN"
                    clean_df['Category'] = clean_df['Cluster_ID'].apply(lambda x: cluster_names.get(x, "ALGEMEEN"))
                    cat_lookup = dict(zip(clean_df[url_col], clean_df['Category']))

                    # 4. INTERNE LINKS BEREKENEN
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
                                    'Focus URL': f_url,
                                    'To Hub': cat_lookup.get(t_url, "ALGEMEEN"),
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
        
        # Matrix bouwen
        matrix = pd.crosstab(data['From Hub'], data['To Hub'])
        
        # Sorteren op Totaal (Descending)
        row_order = matrix.sum(axis=1).sort_values(ascending=False).index
        col_order = matrix.sum(axis=0).sort_values(ascending=False).index
        matrix = matrix.reindex(index=row_order, columns=col_order, fill_value=0)

        st.divider()
        st.subheader("📊 Cross-Linking Matrix (Intensity)")
        st.info("💡 Klik op een rij om de details te zien. De matrix is gesorteerd op volume.")

        max_val = matrix.values.max() if matrix.values.max() > 0 else 1
        def style_matrix_cells(val):
            if val == 0:
                return 'background-color: #0a0a0a; color: #222222; text-align: center;'
            else:
                intensity = 0.2 + 0.8 * (val / max_val)
                return f'background-color: rgba(0, 162, 255, {intensity}); color: #ffffff; font-weight: bold; text-align: center;'

        styled_matrix = matrix.style.map(style_matrix_cells)

        st.dataframe(
            styled_matrix,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            key="matrix_selector"
        )

        selection = st.session_state.get("matrix_selector")
        if selection and selection.get("selection", {}).get("rows"):
            selected_idx = selection["selection"]["rows"][0]
            f_cat = matrix.index[selected_idx]
            
            st.markdown(f"### 🎯 Uitgaande links vanuit: `{f_cat}`")
            filtered = data[data['From Hub'] == f_cat]
            display_filtered = filtered[['Focus URL', 'To Hub', 'Target URL', 'Score']].sort_values(by=['Focus URL', 'Score'], ascending=[True, False]).copy()
            display_filtered.loc[display_filtered.duplicated('Focus URL'), 'Focus URL'] = ""
            
            st.dataframe(
                display_filtered.style.map(color_score, subset=['Score']),
                use_container_width=True,
                hide_index=True,
                column_config={"Score": st.column_config.NumberColumn(format="%d%%")}
            )

        # ========================================================
        # 7. TOPIC HUBS OVERZICHT (Indeling op Sterkte)
        # ========================================================
        st.divider()
        st.subheader("🏗️ Topic Hubs Overzicht")

        # Bereken de gemiddelde scores per hub
        hub_stats = data.groupby('From Hub')['Score'].mean().sort_values(ascending=False)

        # Maak de drie tabbladen aan
        tab_strong, tab_avg, tab_weak = st.tabs([
            "🟢 Sterk (>= 85%)", 
            "🟡 Gemiddeld (70-84%)", 
            "🔴 Zwak (< 70%)"
        ])

        # Hulpfunctie om de hubs binnen een tabblad te tonen
        def render_hub_group(hubs_series):
            if hubs_series.empty:
                st.info("Geen hubs gevonden voor deze categorie.")
            else:
                for hub, avg_score in hubs_series.items():
                    hub_df = data[data['From Hub'] == hub]
                    with st.expander(f"📁 HUB: {hub} ({round(avg_score)}%)"):
                        # Sorteer op Focus URL en score, verberg duplicaten
                        display_hub = hub_df[['Focus URL', 'To Hub', 'Target URL', 'Score']].sort_values(by=['Focus URL', 'Score'], ascending=[True, False]).copy()
                        display_hub.loc[display_hub.duplicated('Focus URL'), 'Focus URL'] = ""
                        
                        st.dataframe(
                            display_hub.style.map(color_score, subset=['Score']),
                            use_container_width=True,
                            hide_index=True,
                            column_config={"Score": st.column_config.NumberColumn(format="%d%%")}
                        )

        # Deel de hubs in op basis van de kleurenschaal/gemiddelde score
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
        
        # Maak Focus URL leeg voor herhalende waarden
        export_df.loc[export_df.duplicated(subset=['From Hub', 'Focus URL']), 'Focus URL'] = ""
        # Maak From Hub leeg voor herhalende waarden (toont de hub slechts 1x per groep)
        export_df.loc[export_df.duplicated(subset=['From Hub']), 'From Hub'] = ""
        
        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False, sep=';')
        
        st.download_button(
            label="📥 Download Resultaten (CSV)",
            data=csv_buffer.getvalue(),
            file_name="seo_internal_links_matrix.csv",
            mime="text/csv"
        )
