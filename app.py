import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# -----------------------------------------------------------------------------
# CONFIGURATION DE LA PAGE & THÈME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AutoMPG • Studio Analytics & ML Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CSS PERSONNALISÉ ULTRA-MODERNE (DARK GLASSMORPHISM & NEON ACCENTS)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* Fond principal avec dégradé subtil */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #111827 0%, #0B0F19 55%, #05070d 100%) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: #E2E8F0 !important;
    }

    /* Personnalisation de la barre latérale */
    [data-testid="stSidebar"] {
        background: rgba(13, 18, 30, 0.85) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    /* Cartes Glassmorphism */
    .glass-card {
        background: rgba(22, 30, 49, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }

    /* En-tête Hero */
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.35);
        color: #A5B4FC;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        line-height: 1.15;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #FFFFFF 0%, #E2E8F0 50%, #94A3B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }

    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 24px;
    }

    /* Banner d'info technologique */
    .tech-banner {
        background: linear-gradient(90deg, rgba(56, 189, 248, 0.1) 0%, rgba(99, 102, 241, 0.1) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 12px;
        padding: 14px 20px;
        color: #BAE6FD;
        font-size: 0.92rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 28px;
    }

    /* Titres de section */
    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.01em;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 10px;
        margin-bottom: 16px;
    }

    .section-title::before {
        content: '';
        display: inline-block;
        width: 4px;
        height: 20px;
        background: linear-gradient(180deg, #38BDF8, #6366F1);
        border-radius: 4px;
    }

    /* Cartes d'indicateurs (KPI) */
    .kpi-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.85));
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 22px;
        position: relative;
        overflow: hidden;
        transition: transform 0.25s ease, border-color 0.25s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
    }
    .kpi-cyan::before { background: linear-gradient(90deg, #06B6D4, #3B82F6); }
    .kpi-emerald::before { background: linear-gradient(90deg, #10B981, #14B8A6); }
    .kpi-amber::before { background: linear-gradient(90deg, #F59E0B, #EF4444); }
    .kpi-purple::before { background: linear-gradient(90deg, #8B5CF6, #EC4899); }

    .kpi-label {
        color: #94A3B8;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 8px;
    }
    .kpi-value {
        color: #FFFFFF;
        font-size: 2.3rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: -0.02em;
        line-height: 1;
        margin-bottom: 6px;
    }
    .kpi-sub {
        color: #64748B;
        font-size: 0.8rem;
    }

    /* Boutons personnalisés avec dégradé vibrant */
    .stButton > button {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        font-size: 1rem !important;
        box-shadow: 0 6px 20px -4px rgba(99, 102, 241, 0.6) !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 25px -4px rgba(99, 102, 241, 0.8) !important;
        border-color: rgba(255, 255, 255, 0.3) !important;
    }

    /* Dataframes et tableaux Streamlit */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }

    /* Custom form styling */
    [data-testid="stForm"] {
        background: rgba(22, 30, 49, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        padding: 24px !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# -----------------------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    df['year'] = df['model_year'] + 1900
    df['consumption_l_100km'] = (235.214583 / df['mpg']).round(2)
    df['weight_kg'] = (df['weight'] * 0.45359237).round(2)
    df['mpg'] = df['mpg'].round(2)
    return df

@st.cache_resource
def train_random_forest(data):
    df_ml = data.copy()
    features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin']
    target = 'mpg'

    X = df_ml[features]
    y = df_ml[target]

    numeric_features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']
    categorical_features = ['origin']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }

    rf = model.named_steps['regressor']
    cat_encoder = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    cat_names = list(cat_encoder.get_feature_names_out(categorical_features))
    feature_names = numeric_features + cat_names
    importances = pd.DataFrame({
        'Caractéristique': feature_names,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)

    return model, metrics, importances

df_raw = load_data()

# -----------------------------------------------------------------------------
# 2. BARRE LATÉRALE : FILTRES & NAVIGATION FUTURISTE
# -----------------------------------------------------------------------------
st.sidebar.markdown("""
<div style="padding: 10px 0 20px 0;">
    <div style="font-size: 1.25rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em;">
        ⚡ TELEMETRY <span style="color: #38BDF8;">HUB</span>
    </div>
    <div style="font-size: 0.8rem; color: #64748B;">Auto MPG Intelligence Dashboard</div>
</div>
""", unsafe_allow_html=True)

section = st.sidebar.radio(
    "Navigation de l'application",
    ["📊 Analyse descriptive", "🤖 Prédiction ML (Random Forest)"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ **Filtres du Dataset**")

origins = ['Toutes'] + sorted(list(df_raw['origin'].dropna().unique()))
selected_origin = st.sidebar.selectbox("Origine géographique", origins, index=0)

all_cylinders = sorted(list(df_raw['cylinders'].unique()))
selected_cylinders = st.sidebar.multiselect(
    "Nombre de cylindres",
    options=all_cylinders,
    default=all_cylinders
)

min_year = int(df_raw['year'].min())
max_year = int(df_raw['year'].max())
selected_years = st.sidebar.slider(
    "Plage d'années du modèle",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

min_mpg = float(df_raw['mpg'].min())
max_mpg = float(df_raw['mpg'].max())
selected_min_mpg = st.sidebar.slider(
    "Efficacité minimale (mpg)",
    min_value=min_mpg,
    max_value=max_mpg,
    value=min_mpg,
    step=0.1
)

st.sidebar.markdown("""
<div style="margin-top: 30px; padding: 14px; background: rgba(255,255,255,0.03); border-radius: 12px; border: 1px solid rgba(255,255,255,0.06); font-size: 0.78rem; color: #64748B;">
    💡 <b>Astuce télémétrique :</b> 1 mpg (US) équivaut à 235.215 / L/100 km. Plus le mpg est élevé, plus l'empreinte énergétique est faible.
</div>
""", unsafe_allow_html=True)

# Application des filtres
df_filtered = df_raw.copy()

if selected_origin != 'Toutes':
    df_filtered = df_filtered[df_filtered['origin'] == selected_origin]

if selected_cylinders:
    df_filtered = df_filtered[df_filtered['cylinders'].isin(selected_cylinders)]
else:
    df_filtered = df_filtered.iloc[0:0]

df_filtered = df_filtered[
    (df_filtered['year'] >= selected_years[0]) &
    (df_filtered['year'] <= selected_years[1]) &
    (df_filtered['mpg'] >= selected_min_mpg)
]

display_cols = [
    'name', 'origin', 'year', 'cylinders', 'mpg',
    'consumption_l_100km', 'horsepower', 'weight_kg', 'acceleration'
]

# Palette de couleurs coordonnées cyber-modernes
COLOR_MAP = {
    'europe': '#6366F1',  # Electric Indigo
    'japan': '#06B6D4',   # Cyber Cyan
    'usa': '#F43F5E'      # Neon Coral
}

# -----------------------------------------------------------------------------
# 3. INTERFACE : ANALYSE DESCRIPTIVE
# -----------------------------------------------------------------------------
if section == "📊 Analyse descriptive":
    st.markdown('<div class="hero-badge">⚡ TELEMETRY EXPLORER</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">🚗 Exploration des Données Automobiles</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Dataset historique Auto MPG — Analyse exploratoire, filtrage multidimensionnel & visualisations comparatives.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="tech-banner">
        <span>💡</span>
        <span><b>Guide de lecture :</b> Un <b>mpg élevé</b> traduit une meilleure efficacité énergétique, tandis qu'un <b>L/100 km faible</b> signifie une consommation moindre. Analyse descriptive en temps réel.</span>
    </div>
    """, unsafe_allow_html=True)

    # 1 — Indicateurs et données filtrées
    st.markdown(f'<div class="section-title">1 — Indicateurs Clés & Données Filtrées ({len(df_filtered)} / {len(df_raw)} véhicules)</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    if len(df_filtered) > 0:
        avg_mpg = df_filtered['mpg'].mean()
        avg_cons = df_filtered['consumption_l_100km'].mean()
        avg_hp = df_filtered['horsepower'].dropna().mean()

        with c1:
            st.markdown(f"""
            <div class="kpi-card kpi-cyan">
                <div class="kpi-label">Efficacité Moyenne</div>
                <div class="kpi-value">{avg_mpg:.1f} <span style="font-size: 1.2rem; color: #38BDF8;">mpg</span></div>
                <div class="kpi-sub">Moyenne de l'échantillon sélectionné</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="kpi-card kpi-emerald">
                <div class="kpi-label">Consommation Moyenne</div>
                <div class="kpi-value">{avg_cons:.1f} <span style="font-size: 1.2rem; color: #34D399;">L/100 km</span></div>
                <div class="kpi-sub">Moyenne des conversions individuelles</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            hp_txt = f"{avg_hp:.1f}" if pd.notnull(avg_hp) else "N/A"
            st.markdown(f"""
            <div class="kpi-card kpi-amber">
                <div class="kpi-label">Puissance Moyenne</div>
                <div class="kpi-value">{hp_txt} <span style="font-size: 1.2rem; color: #FBBF24;">hp</span></div>
                <div class="kpi-sub">Valeurs manquantes ignorées</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Aucun véhicule ne correspond à ces critères de filtrage.")

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    st.dataframe(
        df_filtered[display_cols].reset_index(drop=True),
        height=300
    )

    # 2 — Choisir un véhicule
    st.markdown('<div class="section-title">2 — Fiche Technique d\'un Véhicule</div>', unsafe_allow_html=True)
    if len(df_filtered) > 0:
        vehicle_options = [
            f"{row['name']} — {row['year']} (ligne {idx + 1})"
            for idx, row in df_filtered.iterrows()
        ]
        selected_vehicle_str = st.selectbox("Sélectionnez un modèle pour inspecter ses spécifications :", vehicle_options, index=0)
        selected_row_idx = vehicle_options.index(selected_vehicle_str)
        selected_vehicle = df_filtered.iloc[[selected_row_idx]][display_cols]
        st.dataframe(selected_vehicle)
    else:
        st.info("Sélectionnez des filtres valides pour voir la fiche d'un véhicule.")

    # 3 — Comparaisons graphiques
    st.markdown('<div class="section-title">3 — Comparaisons Graphiques & Corrélations</div>', unsafe_allow_html=True)

    if len(df_filtered) > 0:
        chart_col1, chart_col2 = st.columns(2)

        # A. Nombre d'observations par origine (Bar chart)
        with chart_col1:
            st.markdown("##### 📍 **Nombre d'observations par origine**")
            origin_counts = df_filtered['origin'].value_counts().reset_index()
            origin_counts.columns = ['Origine', 'Nombre']
            origin_order = ['europe', 'japan', 'usa']
            origin_counts['Origine'] = pd.Categorical(origin_counts['Origine'], categories=origin_order, ordered=True)
            origin_counts = origin_counts.sort_values('Origine').dropna()

            fig_bar = px.bar(
                origin_counts,
                x='Origine',
                y='Nombre',
                color='Origine',
                color_discrete_map=COLOR_MAP,
                category_orders={'Origine': origin_order}
            )
            fig_bar.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis=dict(showgrid=False, title=dict(font=dict(color="#94A3B8"))),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)', title=dict(font=dict(color="#94A3B8"))),
                height=350,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_bar)

        # C. Efficacité moyenne par année de modèle (Line chart)
        with chart_col2:
            st.markdown("##### 📈 **Évolution de l'efficacité par année**")
            yearly_avg = df_filtered.groupby('year')['mpg'].mean().reset_index()
            yearly_avg.columns = ['Année du modèle', 'Efficacité moyenne (mpg)']

            fig_line = px.line(
                yearly_avg,
                x='Année du modèle',
                y='Efficacité moyenne (mpg)',
                markers=True
            )
            fig_line.update_traces(
                line=dict(color='#38BDF8', width=3),
                marker=dict(size=8, color='#6366F1', line=dict(width=2, color='#FFFFFF')),
                fill='tozeroy',
                fillcolor='rgba(56, 189, 248, 0.08)'
            )
            fig_line.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)', dtick=2),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)'),
                height=350
            )
            st.plotly_chart(fig_line)

        # B. Relation entre Poids et Efficacité énergétique (Scatter plot)
        st.markdown("##### 🔬 **Relation entre Poids (kg) et Consommation / Efficacité (mpg)**")
        fig_scatter = px.scatter(
            df_filtered,
            x='weight_kg',
            y='mpg',
            color='origin',
            color_discrete_map=COLOR_MAP,
            labels={'weight_kg': 'Poids du véhicule (kg)', 'mpg': 'Efficacité (mpg)', 'origin': 'Origine'},
            hover_name='name',
            hover_data={'year': True, 'horsepower': True, 'consumption_l_100km': True}
        )
        fig_scatter.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=1, color='rgba(255,255,255,0.3)')))
        fig_scatter.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)'),
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_scatter)

# -----------------------------------------------------------------------------
# 4. INTERFACE : MODÈLE ML (RANDOM FOREST REGRESSOR)
# -----------------------------------------------------------------------------
else:
    st.markdown('<div class="hero-badge">🤖 NEURAL REVOLUTION</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Prédiction de Consommation & Random Forest</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Modèle prédictif supervisé estimant l\'efficience énergétique à partir des spécifications mécaniques du véhicule.</div>', unsafe_allow_html=True)

    model, metrics, importances = train_random_forest(df_raw)

    # Cartes de performance du modèle
    st.markdown('<div class="section-title">Performances du Modèle (Test Set 20%)</div>', unsafe_allow_html=True)
    pm1, pm2, pm3 = st.columns(3)

    with pm1:
        st.markdown(f"""
        <div class="kpi-card kpi-purple">
            <div class="kpi-label">Coefficient de Détermination (R²)</div>
            <div class="kpi-value">{metrics['r2'] * 100:.1f} <span style="font-size: 1.2rem; color: #C084FC;">%</span></div>
            <div class="kpi-sub">Excellente corrélation et pouvoir prédictif</div>
        </div>
        """, unsafe_allow_html=True)

    with pm2:
        st.markdown(f"""
        <div class="kpi-card kpi-cyan">
            <div class="kpi-label">Erreur Absolue Moyenne (MAE)</div>
            <div class="kpi-value">±{metrics['mae']:.2f} <span style="font-size: 1.2rem; color: #38BDF8;">mpg</span></div>
            <div class="kpi-sub">Écart moyen de prédiction</div>
        </div>
        """, unsafe_allow_html=True)

    with pm3:
        st.markdown(f"""
        <div class="kpi-card kpi-emerald">
            <div class="kpi-label">Erreur Quadratique (RMSE)</div>
            <div class="kpi-value">{metrics['rmse']:.2f} <span style="font-size: 1.2rem; color: #34D399;">mpg</span></div>
            <div class="kpi-sub">Pénalisation des grands écarts</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Formulaire de simulation
    st.markdown('<div class="section-title">Paramètres de Télémesure du Véhicule</div>', unsafe_allow_html=True)

    with st.form("ml_prediction_form"):
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("<span style='color: #94A3B8; font-weight:600; font-size: 0.9rem;'>Moteur & Origine</span>", unsafe_allow_html=True)
            p_cylinders = st.selectbox("Nombre de cylindres", options=[3, 4, 5, 6, 8], index=2)
            p_displacement = st.number_input("Cylindrée (displacement en cu. in.)", min_value=50.0, max_value=500.0, value=200.0, step=5.0)
            p_horsepower = st.number_input("Puissance moteur (horsepower en hp)", min_value=40.0, max_value=300.0, value=100.0, step=1.0)
            p_origin = st.selectbox("Zone géographique d'origine", options=['usa', 'japan', 'europe'], index=0)

        with col_right:
            st.markdown("<span style='color: #94A3B8; font-weight:600; font-size: 0.9rem;'>Châssis & Époque</span>", unsafe_allow_html=True)
            p_weight_kg = st.number_input("Masse du véhicule (en kilogrammes)", min_value=600.0, max_value=2600.0, value=1400.0, step=10.0)
            p_weight_lbs = p_weight_kg / 0.45359237

            p_acceleration = st.number_input("Accélération (0-60 mph en secondes)", min_value=5.0, max_value=30.0, value=15.0, step=0.5)
            p_year = st.slider("Année de mise en circulation", min_value=1970, max_value=1982, value=1978)
            p_model_year = p_year - 1900

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("⚡ Calculer l'Estimation Neuronale")

    if submit_btn:
        input_data = pd.DataFrame([{
            'cylinders': p_cylinders,
            'displacement': p_displacement,
            'horsepower': p_horsepower,
            'weight': p_weight_lbs,
            'acceleration': p_acceleration,
            'model_year': p_model_year,
            'origin': p_origin
        }])

        pred_mpg = model.predict(input_data)[0]
        pred_l_100km = 235.214583 / pred_mpg

        st.markdown('<div class="section-title">Résultats de la Simulation</div>', unsafe_allow_html=True)
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.markdown(f"""
            <div class="kpi-card kpi-cyan">
                <div class="kpi-label">Efficacité Énergétique Prédite</div>
                <div class="kpi-value">{pred_mpg:.2f} <span style="font-size: 1.3rem; color: #38BDF8;">mpg</span></div>
                <div class="kpi-sub">Miles per gallon (norme US)</div>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown(f"""
            <div class="kpi-card kpi-emerald">
                <div class="kpi-label">Consommation Équivalente</div>
                <div class="kpi-value">{pred_l_100km:.2f} <span style="font-size: 1.3rem; color: #34D399;">L/100 km</span></div>
                <div class="kpi-sub">Standard européen (235.215 / mpg)</div>
            </div>
            """, unsafe_allow_html=True)

    # Importance des variables (Feature Importance)
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Importance des Variables (Random Forest Feature Importance)</div>', unsafe_allow_html=True)

    fig_imp = px.bar(
        importances,
        x='Importance',
        y='Caractéristique',
        orientation='h',
        color='Importance',
        color_continuous_scale=['#1E293B', '#38BDF8', '#6366F1', '#EC4899']
    )
    fig_imp.update_layout(
        template='plotly_dark',
        yaxis=dict(autorange="reversed", title=""),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)', title="Poids dans la décision du modèle"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20),
        height=340
    )
    st.plotly_chart(fig_imp)
