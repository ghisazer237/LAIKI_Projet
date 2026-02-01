import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import os
import joblib
from io import BytesIO

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title='Diamonds - Decision Tree', layout='wide')

@st.cache_data
def load_data(path='data/diamonds.csv'):
    # Try local file, otherwise load from seaborn and save
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = sns.load_dataset('diamonds')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
    return df


def build_preprocessor(df, numeric_features=None, categorical_features=None):
    if numeric_features is None:
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if categorical_features is None:
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = StandardScaler()
    # Use `sparse_output` for newer scikit-learn versions; fall back to `sparse` for older versions
    try:
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ],
        remainder='drop',
        verbose_feature_names_out=False,
    )
    return preprocessor


def build_pipeline(preprocessor, random_state=42, **dt_params):
    model = DecisionTreeRegressor(random_state=random_state, **dt_params)
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    return pipe


def evaluate_model(pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}


def get_feature_names(preprocessor, df):
    # scikit-learn >=1.0 supports get_feature_names_out on ColumnTransformer
    try:
        feature_names = preprocessor.get_feature_names_out(df.columns.tolist())
    except Exception:
        # fallback: approximate
        num = preprocessor.transformers_[0][2]
        cat = preprocessor.transformers_[1][2]
        cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat)
        feature_names = list(num) + list(cat_names)
    return feature_names


# ----------------- Streamlit App -----------------

def main():
    # --- Custom styling for better UX ---
    st.markdown("""
    <style>
    .app-header {text-align: left; padding: 0.5rem 0;}
    .big-title {font-size:34px; font-weight:700;}
    .subtitle {color: #555; margin-top:-10px; margin-bottom:18px}
    .card {background:#f8f9fa; padding:10px; border-radius:8px;}
    </style>
    """, unsafe_allow_html=True)

    col_header1, col_header2 = st.columns([6,1])
    with col_header1:
        st.markdown('<div class="app-header">', unsafe_allow_html=True)
        st.markdown('<div class="big-title">Diamonds Price Predictor</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Prédiction d\'une colonne cible avec Decision Tree — utilisation du fichier de base (data/diamonds.csv).</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_header2:
        st.image('https://raw.githubusercontent.com/encharm/Font-Awesome-SVG-PNG/master/black/png/64/diamond.png', width=48)

    # Sidebar organized with expanders
    st.sidebar.header('Contrôles')
    with st.sidebar.expander('Données'):
        st.write('Le dataset utilisé est le fichier de base : `data/diamonds.csv`')
        if st.button('Recharger le dataset'):
            # Vider le cache pour forcer le rechargement, puis relancer l'app
            st.cache_data.clear()
            st.experimental_rerun()

    with st.sidebar.expander('Configuration du modèle'):
        test_size = st.slider('Taille du jeu de test', 0.05, 0.5, 0.2, help='Fraction du dataset à utiliser pour le test')
        random_state = st.number_input('Random state', value=42, help='Fixez pour reproductibilité')

    with st.sidebar.expander('Optimisation'):
        n_iter = st.number_input('n_iter (RandomizedSearch)', min_value=10, max_value=200, value=30, step=10)
        cv = st.number_input('cv (cross-val)', min_value=2, max_value=10, value=5)

    # session state initialization
    if 'last_model' not in st.session_state:
        st.session_state.last_model = None
    if 'last_metrics' not in st.session_state:
        st.session_state.last_metrics = None

    # Utilisation du dataset de base (plus d'upload utilisateur)
    df = load_data()
    st.sidebar.info(f"Dataset chargé depuis `data/diamonds.csv` (lignes: {df.shape[0]}, colonnes: {df.shape[1]})")

    st.sidebar.write('Taille du dataset: ', df.shape)
    if st.sidebar.checkbox('Afficher les 5 premières lignes'):
        st.dataframe(df.head())

    # Sélection de la colonne cible (target). Par défaut: 'price' > 'prix' > première colonne
    default_target = 'price' if 'price' in df.columns else ('prix' if 'prix' in df.columns else df.columns[0])
    target = st.sidebar.selectbox('Cible (target) - colonne à prédire', options=df.columns.tolist(), index=int(df.columns.get_loc(default_target)))
    if target is None or target == '':
        st.error('Veuillez sélectionner la colonne cible.')
        st.stop()

    features = st.sidebar.multiselect('Colonnes utilisées comme features (laisser vide pour toutes sauf target)', options=[c for c in df.columns if c != target])
    if not features:
        features = [c for c in df.columns if c != target]

    st.header('Préparation des données')
    X = df[features]
    y = df[target]

    st.write('Features sélectionnées:', features)

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Dataset overview card
    with st.container():
        st.markdown('## Aperçu du dataset')
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric('Lignes', X.shape[0])
        with c2:
            st.metric('Colonnes', X.shape[1])
        with c3:
            st.metric('Colonnes cat.', len(categorical_features))

        col1, col2 = st.columns([2,1])
        with col1:
            st.subheader(f'Distribution de `{target}`')
            st.bar_chart(df[target].value_counts().head(50))
        with col2:
            st.subheader('Statistiques rapides')
            st.write(df[target].describe().to_frame())

        with st.expander('Visualisations avancées'):
            st.subheader('Visualisations avancées')
            plot_type = st.selectbox('Type de graphique', options=['Correlation heatmap', 'Scatter matrix (pairplot)', 'Distribution', 'Boxplot'])
            try:
                import plotly.express as px
                import plotly.graph_objects as go
                import matplotlib.pyplot as plt

                if plot_type == 'Correlation heatmap':
                    corr_df = X[numeric_features].corr()
                    if corr_df.empty or corr_df.shape[0] < 2:
                        st.info('Pas assez de colonnes numériques pour afficher une heatmap (au moins 2 sont requises).')
                    elif corr_df.isnull().all().all():
                        st.info("La matrice de corrélation contient uniquement des valeurs manquantes ou constantes; impossible d'afficher la heatmap.")
                    else:
                        fig = px.imshow(corr_df, color_continuous_scale='RdBu', zmin=-1, zmax=1, template='plotly_dark')
                        fig.update_layout(margin=dict(l=40,r=20,t=30,b=20), height=460)
                        st.plotly_chart(fig, use_container_width=True)

                elif plot_type == 'Scatter matrix (pairplot)':
                    if len(numeric_features) < 2:
                        st.info('Pas assez de colonnes numériques pour afficher un scatter matrix (au moins 2 sont requises).')
                    else:
                        cols = st.multiselect('Colonnes pour scatter matrix', options=numeric_features, default=numeric_features[:4])
                        if cols:
                            fig = px.scatter_matrix(X[cols], dimensions=cols, template='plotly_dark')
                            fig.update_traces(diagonal_visible=False)
                            fig.update_layout(height=600, margin=dict(t=20))
                            st.plotly_chart(fig, use_container_width=True)

                elif plot_type == 'Distribution':
                    if not numeric_features:
                        st.info('Aucune colonne numérique disponible pour les distributions.')
                    else:
                        col = st.selectbox('Choisir une colonne numérique', options=numeric_features)
                        if col:
                            fig = px.histogram(X, x=col, nbins=60, template='plotly_dark', color_discrete_sequence=["#06b6d4"], marginal='violin')
                            fig.update_layout(height=460, margin=dict(t=10))
                            st.plotly_chart(fig, use_container_width=True)

                elif plot_type == 'Boxplot':
                    if not categorical_features:
                        st.info('Aucune colonne catégorielle disponible pour boxplot.')
                    else:
                        cat = st.selectbox('Choisir colonne catégorielle', options=categorical_features)
                        val = st.selectbox('Variable numérique', options=numeric_features)
                        if cat and val:
                            fig = px.box(df, x=cat, y=val, template='plotly_dark', color_discrete_sequence=['#7c3aed'])
                            fig.update_layout(height=460, margin=dict(t=10))
                            st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.write('Impossible d\'afficher les visualisations avancées:', e)

    preprocessor = build_preprocessor(X, numeric_features=numeric_features, categorical_features=categorical_features)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Entraînement baseline')
        with st.form('baseline_form'):
            max_depth = st.number_input('max_depth (None = pas de limite)', min_value=1, max_value=100, value=10)
            min_samples_leaf = st.number_input('min_samples_leaf', min_value=1, max_value=50, value=1)
            submit_baseline = st.form_submit_button('Entraîner Decision Tree (baseline)')
        if submit_baseline:
            with st.spinner('Entraînement...'):
                pipe = build_pipeline(preprocessor, random_state=int(random_state), max_depth=(None if int(max_depth) <= 0 else int(max_depth)), min_samples_leaf=int(min_samples_leaf))
                pipe.fit(X_train, y_train)
                metrics = evaluate_model(pipe, X_test, y_test)
                st.success('Entraînement terminé ✅')
                st.metric('RMSE', f"{metrics['RMSE']:.2f}")
                st.metric('MAE', f"{metrics['MAE']:.2f}")
                st.metric('R2', f"{metrics['R2']:.4f}")

                # Feature importances
                try:
                    feature_names = get_feature_names(pipe.named_steps['preprocessor'], X)
                    importances = pipe.named_steps['model'].feature_importances_
                    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(20)
                    st.subheader('Feature importances')
                    st.bar_chart(fi)
                except Exception as e:
                    st.write('Impossible d\'afficher les importances:', e)

                # Save model to session and provide download
                st.session_state.last_model = pipe
                st.session_state.last_metrics = metrics
                model_path = 'model_baseline.joblib'
                joblib.dump(pipe, model_path)
                with open(model_path, 'rb') as f:
                    st.download_button('Télécharger le modèle (baseline)', f, file_name='model_baseline.joblib')

    with col2:
        st.subheader('Optimisation (RandomizedSearchCV)')
        with st.form('tuning_form'):
            n_iter_local = st.number_input('n_iter', min_value=10, max_value=200, value=int(n_iter), step=10)
            cv_local = st.number_input('cv', min_value=2, max_value=10, value=int(cv))
            submit_tune = st.form_submit_button('Lancer l\'optimisation')

        if submit_tune:
            with st.spinner('Recherche d\'hyperparamètres...'):
                pipe = build_pipeline(preprocessor, random_state=int(random_state))
                param_dist = {
                    'model__max_depth': [None] + list(range(3, 30)),
                    'model__min_samples_split': list(range(2, 20)),
                    'model__min_samples_leaf': list(range(1, 20)),
                    'model__max_features': [None, 'auto', 'sqrt', 'log2'],
                    'model__criterion': ['squared_error', 'friedman_mse', 'absolute_error']
                }
                search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=int(n_iter_local), cv=int(cv_local), n_jobs=-1, scoring='neg_root_mean_squared_error', random_state=42, verbose=0)
                search.fit(X_train, y_train)
                best = search.best_estimator_
                st.success('Optimisation terminée ✅')
                st.write('Meilleurs paramètres:')
                st.json(search.best_params_)
                metrics = evaluate_model(best, X_test, y_test)
                st.metric('RMSE', f"{metrics['RMSE']:.2f}")
                st.metric('MAE', f"{metrics['MAE']:.2f}")
                st.metric('R2', f"{metrics['R2']:.4f}")

                # Feature importances
                try:
                    feature_names = get_feature_names(best.named_steps['preprocessor'], X)
                    importances = best.named_steps['model'].feature_importances_
                    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(20)
                    st.subheader('Feature importances (tuned)')
                    st.bar_chart(fi)
                except Exception as e:
                    st.write('Impossible d\'afficher les importances:', e)

                st.session_state.last_model = best
                st.session_state.last_metrics = metrics
                model_path = 'model_tuned.joblib'
                joblib.dump(best, model_path)
                with open(model_path, 'rb') as f:
                    st.download_button('Télécharger le modèle (tuned)', f, file_name='model_tuned.joblib')

    st.sidebar.markdown('---')
    st.sidebar.markdown('Copyright ©LPGentreprises')


if __name__ == '__main__':
    main()
