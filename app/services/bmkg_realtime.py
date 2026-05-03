import requests

# fetch data from BMKG API real-time endpoint
def fetch_bmkg_data():
    bmkg_url = "https://data.bmkg.go.id/DataMKG/TEWS/gempadirasakan.json"
    response = requests.get(bmkg_url, timeout=10)
    
    if response.status_code != 200:
        raise RuntimeError("Gagal fetch BMKG")
    
    return response.json()