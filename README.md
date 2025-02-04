# Classification des Mouvements de Prix des Cryptomonnaies : Une Approche d'Apprentissage Automatique

Ce projet utilise des techniques d'apprentissage automatique pour prédire l'évolution des prix de 6 cryptomonnaies en utilisant une approche par classification. Les cryptomonnaies prises en charge sont :

- ETH (Ethereum)
- XRP (Ripple)
- LINK (Chainlink)
- SOL (Solana)
- BTC (Bitcoin)

# Description
Le projet combine l'analyse technique, le web scraping et l'apprentissage automatique pour prédire les mouvements de prix des cryptomonnaies. Nous avons utilisé des techniques avancées pour récupérer et analyser des données sur six cryptomonnaies populaires : ETH (Ethereum), XRP (Ripple), LINK (Chainlink), SOL (Solana), BTC (Bitcoin) et BNB (Binance Coin).

### Étape 1 : Web Scraping et Collecte de Données
  
Nous avons réalisé un processus de web scraping sur un site spécialisé afin de collecter des données historiques de prix et d'autres informations pertinentes pour chaque cryptomonnaie. Le scraping nous a permis d'extraire des informations telles que les prix, les volumes de transactions, et d'autres données spécifiques nécessaires à l'analyse technique. Ces données ont ensuite été nettoyées et préparées pour l'analyse.

### Étape 2 : Calcul des Indicateurs Techniques
  
Pour améliorer la qualité de nos prédictions, nous avons calculé plusieurs indicateurs techniques qui sont couramment utilisés dans le trading de cryptomonnaies. Ces indicateurs nous aident à analyser la tendance des prix et à anticiper leurs mouvements futurs. Les indicateurs calculés sont les suivants :

- **RSI (Relative Strength Index)** : Un indicateur de momentum qui mesure la vitesse et le changement des mouvements de prix pour identifier les conditions de surachat ou de survente.
- **Stochastic Oscillator** : Un indicateur de momentum qui compare le prix de clôture d'une cryptomonnaie par rapport à son prix sur une période donnée.
- **Williams %R** : Un autre indicateur de momentum similaire au Stochastic Oscillator, utilisé pour identifier les conditions de surachat et de survente.
- **MACD (Moving Average Convergence Divergence)** : Un indicateur qui suit les tendances, mesurant la relation entre deux moyennes mobiles.
- **PROC (Percentage Rate of Change)** : Mesure la variation en pourcentage du prix sur une période donnée, utile pour identifier les tendances de prix.
- **OBV (On-Balance Volume)** : Un indicateur de volume qui estime la pression d'achat et de vente en fonction des variations de prix et des volumes.

Ces indicateurs ont été intégrés aux données de prix pour créer des ensembles de caractéristiques robustes, qui ont été utilisés dans les modèles de classification.

### Étape 3 : Modélisation et Prédiction avec Random Forest et XGBoost
Une fois les données préparées, nous avons utilisé deux modèles de classification puissants pour prédire la direction future des prix des cryptomonnaies : Random Forest et XGBoost. Ces modèles sont particulièrement efficaces pour les tâches de classification et peuvent gérer des jeux de données avec une grande quantité de variables.

- **Random Forest** a été utilisé pour entraîner plusieurs arbres de décision, permettant de prédire si le prix d'une cryptomonnaie va augmenter ou diminuer sur une période donnée.
- **XGBoost** a été utilisé pour optimiser les performances de la prédiction en combinant plusieurs arbres de décision de manière séquentielle pour réduire les erreurs et améliorer la précision des prévisions.
Nous avons appliqué ces modèles sur plusieurs fenêtres de trading, ce qui nous a permis de réaliser des prédictions à court, moyen et long terme pour chaque cryptomonnaie.

### Étape 4 : API FastAPI (Backend)
Le projet inclut également une API FastAPI qui permet aux utilisateurs d'interagir avec différentes parties du système. Cette API expose plusieurs points de terminaison (endpoints) qui permettent d'exécuter des tâches automatisées et d’obtenir des prédictions basées sur les modèles de machine learning. Les endpoints principaux sont les suivants :

- **Index Endpoint**: Fournit une explication concise de l'application et de ses capacités, ainsi qu'un lien vers la documentation de l'API et la liste des endpoints disponibles.
- **Health Check Endpoint**: Permet de vérifier la disponibilité de l'API et l'état du système.
- **Web Scraping Endpoint**: Permet d'exécuter des tâches de web scraping en fonction des paramètres fournis. Ce point de terminaison permet de récupérer des données à jour sur les cryptomonnaies.
- **Model Inference Endpoint**: Exécute des prédictions à l’aide des modèles de machine learning (Random Forest et XGBoost). Ce point de terminaison permet aux utilisateurs d’obtenir des prédictions sur la direction future des prix des cryptomonnaies.

Pour exécuter le script CryptoAPI.py, utilisez la commande suivante dans votre terminal :
   ```bash
   uvicorn CryptoAPI:app --reload
   ```

En résumé, ce projet offre une solution complète pour prédire les mouvements de prix des cryptomonnaies en utilisant des méthodes avancées de machine learning, d'analyse technique, de web scraping, et d'API pour l'interaction et l'automatisation des tâches. Grâce à cette approche intégrée, nous pouvons fournir des prédictions précises et en temps réel pour les traders de cryptomonnaies.

# Installation

1. Cloner le dépôt sur votre machine :
   
   ```bash
   git clone https://github.com/votre-username/predict-crypto-prices-using-classification.git
   cd predict-crypto-prices-using-classification
   ```

2. Créer un environnement virtuel :
   #### Sous Windows :
         
   ```bash
   python -m venv env
   ```
   #### Sous macOS/Linux :
         
   ```bash
   python3 -m venv env
   ```
3. Activer l'environnement virtuel :

   #### Sous Windows :
   
   ```bash
   .\env\Scripts\activate
   ```
   #### Sous macOS/Linux :
   
   ```bash
   source env/bin/activate
   ```

4. Installer les dépendances avec pip :
   
   ```bash
   pip install -r requirements.txt
   ```
# Structure des fichiers
``` 
/Data             # Contient les données scrapées
    ├── BNB_data.csv  # Jeu de données pour BNB (Binance Coin)
    ├── BTC_data.csv  # Jeu de données pour BTC (Bitcoin)
    ├── ETH_data.csv  # Jeu de données pour ETH (Ethereum)
    ├── XRP_data.csv  # Jeu de données pour XRP (Ripple)
    ├── LINK_data.csv # Jeu de données pour LINK (Chainlink)
    └── SOL_data.csv  # Jeu de données pour SOL (Solana)

/random_forest_model.pkl  # Modèle Random Forest entraîné
/xgboost_model.pkl        # Modèle XGBoost entraîné

/resume_performances_randomforest.xslx # Comparaison des performances avec random forest
/resume_performances_xgb_classifier.xslx # Comparaison des performances avec xgboost

/full_crypto_prediction.ipynb      # Contient le notebook principal avec tous les plots

/WebScraping.py      # Extraction de données depuis le web
/DataPreprocessor.py # Prétraitement et nettoyage des données
/TechnicalIndicators.py # Calcul des indicateurs techniques (RSI, MACD, etc.)
/CryptoPrediction.py # Logique principale de prédiction et entraînement des modèles
/CryptoAPI.py       # API FastAPI exposant les endpoints pour les prédictions, le scraping, etc.

/README.md           # Documentation du projet
/Requirements.txt    # Liste des dépendances Python
```
# Contributeurs

- **Sara LAVAL-JEANTET**
- **Minh Nhat Thy HUYNH**
- **Salma BENMOUSSA**

Ce projet a été développé dans le cadre du Master MOSEF, à l'université Paris 1 Panthéon Sorbonne.

# 📩 Contact

N'hésitez pas à nous contacter pour toute question :

- sara.jeantet@gmail.com
- hmnthy@gmail.com
- salmabenmoussa103@gmail.com 

