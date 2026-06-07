import httpx

# fetch data from BMKG API real-time endpoint
async def fetch_bmkg_data():
    bmkg_url = "https://data.bmkg.go.id/DataMKG/TEWS/gempadirasakan.json"
    async with httpx.AsyncClient() as client:
        response = await client.get(bmkg_url, timeout=10.0)
    
    if response.status_code != 200:
        raise RuntimeError("Gagal fetch BMKG")
    
    return response.json()