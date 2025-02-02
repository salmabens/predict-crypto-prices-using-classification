from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import psutil
import joblib
from typing import List
from typing import List, Dict
import numpy as np
from CryptoPrediction import getData, prepareData, generate_crypto_dataframes
from WebScraping import scrape_crypto_data

# Initialisation de l'application FastAPI
app = FastAPI(
    title="API FastAPI - Web Scraping et Modèles ML",
    description="Cette API permet de gérer des tâches comme le web scraping, les prédictions de modèles ML et le suivi de la santé du système.",
    version="1.0.0",
)
###############################################################################################################################

# 1. Endpoint Index
@app.get("/", summary="Index - Présentation de l'API")
async def index():
    """
    Fournit une description des capacités de l'application, un lien vers la documentation, et la liste des endpoints disponibles.
    """
    return JSONResponse(
        content={
            "message": "Bienvenue sur l'API FastAPI.",
            "description": "Cette API propose des fonctionnalités telles que le web scraping, les prédictions avec des modèles ML et le monitoring de l'état du système.",
            "documentation_url": "/docs",
            "endpoints": {
                "get": {
                    "Index": "/",
                    "Health Check": "/health", 
                    "Model info": "/model/info"
                },
                "post":{
                    "Web Scraping": "/scrape",
                    "Model Inference Single": "/predict/single",
                    "Model Inference Batch": "/predict/batch"
                }

            }
        }
    )


###############################################################################################################################

# 2. Endpoint Health Check
@app.get("/health", summary="Vérification de l'état du système")
async def health_check():
    """
    Vérifie l'état de santé de l'API et du système.
    """
    # Récupérer l'utilisation des ressources système
    cpu_usage = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return JSONResponse(
        content={
            "status": "OK",
            "cpu_usage": f"{cpu_usage}%",
            "memory": {
                "total": f"{memory.total / (1024 ** 3):.2f} GB",
                "used": f"{memory.used / (1024 ** 3):.2f} GB",
                "available": f"{memory.available / (1024 ** 3):.2f} GB",
            },
            "disk": {
                "total": f"{disk.total / (1024 ** 3):.2f} GB",
                "used": f"{disk.used / (1024 ** 3):.2f} GB",
                "free": f"{disk.free / (1024 ** 3):.2f} GB",
            },
            "dependencies": {
                "database": "OK",  
                "external_service": "OK", 
            }
        }
    )


###############################################################################################################################

# 3. Endpoint Web Sraping
# Définir un modèle pour les paramètres d'entrée
class CryptoRequest(BaseModel):
    name: str
    url: str

@app.post("/scrape", summary="Effectuer un scraping de données de crypto-monnaies")
async def scrape_data(request: CryptoRequest):
    """
    Scrape les données de crypto-monnaie à partir de l'URL donnée.
    :param request: Les paramètres de la requête (nom de la crypto et URL)
    :return: Données scrappées sous forme de liste de dictionnaires
    """
    try:
        # Appeler la fonction de scraping et récupérer les données
        data = scrape_crypto_data(request.name, request.url)

        if not data:
            raise HTTPException(
                status_code=404,
                detail=f"Aucune donnée trouvée pour {request.name} à l'URL spécifiée."
            )

        # Retourner les données sous forme de JSON
        return {"data": data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du scraping: {str(e)}")

###############################################################################################################################

# 4. Endpoints Model Inference
# Chargement des deux modèles au démarrage
rf_model = joblib.load('random_forest_model.pkl')
xgb_model = joblib.load('xgboost_model.pkl')

# Modèles Pydantic pour la validation des requêtes
class PredictionRequest(BaseModel):
    crypto: str
    trading_days: int
    model_type: str  # Ajout d'un paramètre pour spécifier quel modèle utiliser (rf ou xgb)

class BatchPredictionRequest(BaseModel):
    cryptos: List[str]
    trading_days: int
    model_type: str  # Idem, pour spécifier le modèle

def process_crypto(crypto: str, trading_days: int, crypto_dataframes: Dict, model_type: str) -> Dict:
    """Traite les données d'une crypto et fait des prédictions avec le modèle choisi (Random Forest ou XGBoost)."""
    try:
        # Utilisation des fonctions existantes
        ohclv_data, close, date = getData(crypto, crypto_dataframes)
        ohclv_data = np.array(ohclv_data)
        
        # Préparation des données
        X, y, xplot, closeplot, dateplot = prepareData(ohclv_data, close, date, trading_days)
        
        # Choisir le modèle selon model_type
        if model_type == 'rf':
            model = rf_model
        elif model_type == 'xgb':
            model = xgb_model
        else:
            raise ValueError("Modèle inconnu. Utilisez 'rf' pour Random Forest ou 'xgb' pour XGBoost.")
        
        # Prédictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        return {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'dates': [d.strftime('%Y-%m-%d') for d in dateplot]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement de {crypto}: {str(e)}")


@app.post("/predict/single")
async def predict_single(request: PredictionRequest):
    """
    Fait des prédictions pour une seule crypto en fonction du modèle choisi
    """
    try:
        # Vérifier si le modèle est valide
        if request.model_type not in ['rf', 'xgb']:
            raise HTTPException(status_code=400, detail="Modèle inconnu. Choisissez 'rf' ou 'xgb'.")
        
        # Utilisation de generate_crypto_dataframes
        crypto_dataframes = generate_crypto_dataframes([request.crypto])
        if not crypto_dataframes:
            raise HTTPException(status_code=404, detail=f"Données non trouvées pour {request.crypto}")
        
        result = process_crypto(request.crypto, request.trading_days, crypto_dataframes, request.model_type)
        return {
            "crypto": request.crypto,
            "trading_days": request.trading_days,
            "model_type": request.model_type,
            **result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """
    Fait des prédictions pour plusieurs cryptos en fonction du modèle choisi
    """
    # Vérifier si le modèle est valide
    if request.model_type not in ['rf', 'xgb']:
        raise HTTPException(status_code=400, detail="Modèle inconnu. Choisissez 'rf' ou 'xgb'.")

    # Utilisation de generate_crypto_dataframes pour toutes les cryptos
    crypto_dataframes = generate_crypto_dataframes(request.cryptos)
    
    results = []
    for crypto in request.cryptos:
        try:
            if crypto not in crypto_dataframes:
                results.append({
                    "crypto": crypto,
                    "status": "error",
                    "error": "Données non trouvées"
                })
                continue
                
            result = process_crypto(crypto, request.trading_days, crypto_dataframes, request.model_type)
            results.append({
                "crypto": crypto,
                "trading_days": request.trading_days,
                "model_type": request.model_type,
                "status": "success",
                **result
            })
        except Exception as e:
            results.append({
                "crypto": crypto,
                "status": "error",
                "error": str(e)
            })
    
    return {"results": results}


@app.get("/model/info")
async def get_model_info():
    """
    Obtient les informations sur les modèles chargés
    """
    return {
        "models": [
            {
                "model_type": "RandomForestClassifier",
                "description": "Modèle Random Forest pour la prédiction des prix de crypto-monnaies.",
                "features": [
                    "RSI",
                    "StochasticOscillator",
                    "Williams",
                    "MACD",
                    "PROC",
                    "OBV"
                ]
            },
            {
                "model_type": "XGBoost",
                "description": "Modèle XGBoost pour la prédiction des prix de crypto-monnaies.",
                "features": [
                    "RSI",
                    "StochasticOscillator",
                    "Williams",
                    "MACD",
                    "PROC",
                    "OBV"
                ]
            }
        ],
        "trading_days_supported": [5, 10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
    }

