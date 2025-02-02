from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import os
from datetime import datetime, timedelta

def configure_driver():
    chrome_options = Options()
    # Options pour éviter la détection
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-browser-side-navigation")
    chrome_options.add_argument("--disable-gpu")
    
    # Ajout d'un user agent réaliste
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    # Configuration des préférences
    chrome_prefs = {}
    chrome_options.experimental_options["prefs"] = chrome_prefs
    chrome_prefs["profile.default_content_settings"] = {"images": 2}
    chrome_prefs["profile.managed_default_content_settings"] = {"images": 2}

    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )

def get_default_month():
    default_date = datetime.today() - timedelta(days=62)
    return default_date.strftime("%B")

def scrape_crypto_data(crypto_name, url):
    driver = configure_driver()
    driver.maximize_window()
    print(f"Traitement de {crypto_name}...")
    
    try:
        driver.get(url)
        time.sleep(20)  

        # Étape 1 : Vérifier et gérer la bannière des cookies avec plusieurs tentatives
        try:
            # Liste de sélecteurs possibles pour le bouton des cookies
            cookie_selectors = [
                (By.ID, "onetrust-accept-btn-handler"),
                (By.CLASS_NAME, "onetrust-accept-btn-handler"),
                (By.XPATH, "//button[contains(text(), 'Accept All')]"),
                (By.XPATH, "//button[contains(text(), 'Accept all')]"),
                (By.XPATH, "//button[contains(text(), 'Accepter')]"),
                (By.XPATH, "//button[contains(@class, 'cookie-accept')]"),
                (By.CSS_SELECTOR, ".cookie-consent button"),
                (By.CSS_SELECTOR, "#CybotCookiebotDialogBodyButtonAccept"),
                (By.CSS_SELECTOR, ".cmc-cookie-policy-banner__close"),
                (By.XPATH, "//button[contains(@class, 'banner-close-button')]")
            ]

            for selector_type, selector_value in cookie_selectors:
                try:
                    cookie_button = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((selector_type, selector_value))
                    )
                    time.sleep(2)
                    driver.execute_script("arguments[0].click();", cookie_button)
                    print(f"Bannière des cookies acceptée avec le sélecteur: {selector_value}")
                    time.sleep(3)
                    break
                except Exception:
                    continue

        except Exception as e:
            print(f"Tentative de gestion des cookies échouée: {str(e)}")
            print("Continuation du script...")

        # Étape 2 : Cliquer sur le bouton pour ouvrir le calendrier
        calendar_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((
                By.CSS_SELECTOR,
                ".sc-65e7f566-0.eQBACe.BaseButton_base__34gwo.bt-base.BaseButton_t-default__8BIzz.BaseButton_size-md__9TpuT.BaseButton_v-tertiary__AhlyE.BaseButton_vd__gUkWt"
            ))
        )
        calendar_button.click()
        print("Le calendrier a été ouvert avec succès.")
        time.sleep(2)

        # Étape 3 : Sélectionner dynamiquement le mois affiché par défaut
        default_month = get_default_month()
        default_month_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, f"//span[contains(text(), '{default_month}')]"))
        )
        default_month_button.click()
        print(f"Cliqué sur '{default_month}'.")

        jan_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//span[text()='Jan' and contains(@class, ' ')]"))
        )
        jan_button.click()
        print("Cliqué sur 'Jan'.")
        time.sleep(2)

        for _ in range(2):
            january_2024 = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//span[contains(text(), 'January')]"))
            )
            january_2024.click()
            print("Cliqué sur 'January 2024'.")
            time.sleep(1)

        year_2019 = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//span[text()='2019' and contains(@class, ' ')]"))
        )
        year_2019.click()
        print("Sélectionné '2019'.")

        jan_2019 = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//span[contains(@class, 'selected') and text()='Jan']"))
        )
        jan_2019.click()
        print("Cliqué sur 'Jan' pour 2019.")

        day_1_2019 = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((
                By.XPATH,
                "//div[contains(@class, 'react-datepicker__day--001') and contains(@aria-label, 'January 1st, 2019')]"
            ))
        )
        day_1_2019.click()
        print("Cliqué sur le jour '1er janvier 2019'.")
        time.sleep(2)

        january_2019 = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//span[contains(text(), 'January')]"))
        )
        january_2019.click()
        print("Cliqué sur 'January 2019'.")

        dec_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//span[text()='Dec' and contains(@class, ' ')]"))
        )
        dec_button.click()
        print("Cliqué sur 'Dec'.")

        for _ in range(2):
            december_2019 = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//span[contains(text(), 'December')]"))
            )
            december_2019.click()
            print("Cliqué sur 'December 2019'.")
            time.sleep(1)

        year_2024 = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//span[text()='2024' and contains(@class, ' ')]"))
        )
        year_2024.click()
        print("Sélectionné '2024'.")

        dec_2024 = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//span[contains(@class, 'selected') and text()='Dec']"))
        )
        dec_2024.click()
        print("Cliqué sur 'Dec' pour 2024.")

        day_31_2024 = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((
                By.XPATH,
                "//div[contains(@class, 'react-datepicker__day--031') and contains(@aria-label, 'December 31st, 2024')]"
            ))
        )
        day_31_2024.click()
        print("Cliqué sur le jour '31 décembre 2024'.")
        
        continue_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Continue')]"))
        )
        continue_button.click()
        print("Cliqué sur le bouton 'Continue' pour appliquer les dates.")
        time.sleep(5)

        # Étape 4: Load More avec gestion du nombre maximal de clics
        for i in range(30):  # Nombre max de tentatives
            try:
                load_more_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Load More')]"))
                )
                load_more_button.click()
                print(f"Cliqué sur 'Load More' ({i+1}).")
                time.sleep(5)
            except Exception:
                print("Aucun bouton 'Load More' disponible, passage à l'extraction des données.")
                break

        # Étape 5: Extraction des données du tableau
        rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
        data = [[cell.text.strip() for cell in row.find_elements(By.TAG_NAME, "td")] for row in rows]

        df = pd.DataFrame(data, columns=["Date", "Open", "High", "Low", "Close", "Volume", "Market Cap"])
        df["Cryptocurrency"] = crypto_name

        # Assurez-vous que le dossier "Data" existe, sinon créez-le
        data_folder = "Data"
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)  

        file_path = os.path.join(data_folder, f"{crypto_name}_data.csv")  # Chemin complet du fichier
        df.to_csv(file_path, index=False)
        print(f"[{crypto_name}] Données sauvegardées dans {file_path}")   


    except Exception as e:
        print(f"Erreur lors de l'interaction avec {crypto_name}: {e}")
    finally:
        driver.quit()

    return data

if __name__ == "__main__":
    # Liste des URLs et des noms des cryptomonnaies
    cryptos = {
        "BTC": "https://coinmarketcap.com/currencies/bitcoin/historical-data/",
        "ETH": "https://coinmarketcap.com/currencies/ethereum/historical-data/",
        "XRP": "https://coinmarketcap.com/currencies/xrp/historical-data/",
        "BNB": "https://coinmarketcap.com/currencies/bnb/historical-data/",
        "LINK": "https://coinmarketcap.com/currencies/chainlink/historical-data/",
        "SOL": "https://coinmarketcap.com/currencies/solana/historical-data/"
    }
    # Lancer le scraping pour chaque cryptomonnaie
    for crypto_name, url in cryptos.items():
        scrape_crypto_data(crypto_name, url)