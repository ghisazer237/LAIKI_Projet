# Diamonds - Decision Tree (Streamlit)

Application Streamlit pour prédire le prix des diamants (target: `price`) en utilisant un Decision Tree.

Fonctionnalités:
- **Aucune donnée n'est chargée par défaut.** Vous devez uploader votre propre fichier CSV (colonne `price` requise) ou cocher l'option **Charger le jeu d'exemple (Seaborn)** dans la barre latérale pour utiliser un dataset d'exemple.
- Interface améliorée avec formulaires pour l'entraînement et la recherche d'hyperparamètres, visuels d'EDA, et boutons de téléchargement de modèle
- Préprocessing automatique (encodage one-hot, standardisation)
- Entraînement baseline et optimisation d'hyperparamètres via `RandomizedSearchCV`
- Export du modèle entraîné (`.joblib`)

Remarque: l'application vous permet de sélectionner la colonne cible après l'upload. Par défaut elle cherchera `price` puis `prix`, sinon vous pourrez choisir manuellement la colonne à prédire.

Déploiement rapide:
1. Pousser ce repo sur GitHub
2. Déployer sur Streamlit Cloud: créer une nouvelle app et pointer sur `streamlit_app.py`

Alternativement, Heroku/PaaS: créer un `Procfile` et configurer le build.

