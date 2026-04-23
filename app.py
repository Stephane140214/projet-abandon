# ---------- 1. IMPORTER LES BIBLIOTHÈQUES ----------
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ---------- 2. CONFIGURATION DE LA PAGE ----------
st.set_page_config(page_title="Prédiction du Risque d'Abandon", layout="wide")

# ---------- 3. STYLE VISUEL (OPTIONNEL) ----------
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    h1 { color: #1b5e20; }
    .result-risk { color: #c62828; font-weight: bold; font-size: 24px; }
    .result-safe { color: #2e7d32; font-weight: bold; font-size: 24px; }
    </style>
    """, unsafe_allow_html=True)

# ---------- 4. CONSTANTES ----------
MAX_GRADE = 17.96
MAX_RATIO = 10.0
MAX_STUDY = 5.0

# ---------- 5. CHARGEMENT DU MODÈLE ----------
@st.cache_resource
def load_model():
    return joblib.load('model_final_abandon_achi.pkl')

try:
    model = load_model()
except:
    st.error("⚠️ Modèle introuvable. Vérifiez que le fichier 'model_final_abandon_achi.pkl' est dans le même dossier.")
    st.stop()

# ---------- 6. TITRE ----------
st.title("🎓 Prédiction du Risque d'Abandon d'un étudiant")
st.write("Remplissez le profil de l'étudiant ci-dessous, puis cliquez sur « Lancer la Prédiction ».")

# ---------- 7. DEUX COLONNES ----------
col_input, col_res = st.columns([1, 2], gap="large")

# ---------- 8. COLONNE DE GAUCHE : SAISIE ----------
with col_input:
    st.header("📋 Profil de l'étudiant")
    
    age = st.number_input("Âge (entre 15 et 25 ans)", min_value=15, max_value=25, value=18)
    gender = st.selectbox("Sexe", options=["Male", "Female"])
    avg_grade = st.slider("Moyenne générale (sur 20)", min_value=0.0, max_value=20.0, value=11.5, step=0.5)
    absenteeism = st.slider("Taux d'absentéisme (0 = 0%, 0.5 = 50%)", min_value=0.0, max_value=0.5, value=0.1, step=0.01, format="%.2f")
    study_time = st.number_input("Temps d'étude par jour (en heures)", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
    internet = st.radio("Accès à Internet ?", options=["Yes", "No"])
    extra = st.radio("Activités extrascolaires ?", options=["Yes", "No"])

# ---------- 9. CALCUL DES VARIABLES DÉRIVÉES ----------
ratio = (1 - absenteeism) / (absenteeism + 0.1)
score_global = (0.5 * (avg_grade / MAX_GRADE) + 
                0.3 * (ratio / MAX_RATIO) + 
                0.2 * (study_time / MAX_STUDY))

# ---------- 10. COLONNE DE DROITE : RÉSULTATS ----------
with col_res:
    st.header("Résultat de l'analyse")
    
    if st.button("Lancer la Prédiction", type="primary"):
        
        # Préparation des données
        input_df = pd.DataFrame({
            'age': [age],
            'average_grade': [avg_grade],
            'absenteeism_rate': [absenteeism],
            'study_time_hours': [study_time],
            'gender_Male': [1 if gender == "Male" else 0],
            'internet_access_Yes': [1 if internet == "Yes" else 0],
            'extra_activities_Yes': [1 if extra == "Yes" else 0]
        })
        
        input_df['presence_absence_ratio'] = ratio
        input_df['low_grade_flag'] = 1 if avg_grade < 10 else 0
        input_df['global_score'] = score_global
        
        input_df = input_df[model.feature_names_in_]
        
        # Prédiction
        proba_abandon = model.predict_proba(input_df)[0][1]
        
        # Affichage des indicateurs
        st.markdown("**Indicateurs calculés :**")
        st.write(f"Score global : **{score_global:.2f}** (plus il est élevé, mieux c'est)")
        st.write(f"Ratio présence/absence : **{ratio:.2f}** (plus il est élevé, plus l'élève est présent)")
        
        st.markdown("---")
        
        # Diagnostic selon le seuil
        if proba_abandon >= 0.6:
            st.markdown('<span class="result-risk">🚨 RISQUE D\'ABANDON ÉLEVÉ</span>', unsafe_allow_html=True)
            st.error(f"⚠️ Probabilité d'abandon : **{proba_abandon:.1%}**\n\nUne intervention pédagogique immédiate est recommandée.")
        elif proba_abandon >= 0.3:
            st.markdown('<span style="color:#ed6c02; font-weight:bold; font-size:24px;">⚠️ RISQUE MODÉRÉ</span>', unsafe_allow_html=True)
            st.warning(f"⚠️ Probabilité d'abandon : **{proba_abandon:.1%}**\n\nUn suivi régulier est conseillé.")
        else:
            st.markdown('<span class="result-safe">✅ PROFIL STABLE</span>', unsafe_allow_html=True)
            st.success(f"✅ Probabilité d'abandon : **{proba_abandon:.1%}**\n\nL'élève présente de bons indicateurs de réussite.")
        
        # Jauge de risque (version corrigée sans l'erreur)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba_abandon * 100,
            title={"text": "Niveau de risque (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#c62828" if proba_abandon > 0.5 else "#2e7d32"},
                "steps": [
                    {"range": [0, 30], "color": "#a5d6a7"},
                    {"range": [30, 60], "color": "#ffcc80"},
                    {"range": [60, 100], "color": "#ef9a9a"}
                ]
            }
        ))
        
        st.plotly_chart(fig, use_container_width=True)