import requests
import os
import dotenv

url = "https://apis.openapi.sk.com/transit/routes/"
apikey = os.getenv("SK_API_KEY")

payload = {
    "startX": "126.9386",
    "startY": "37.56578",
    "endX": "126.93694",
    "endY": "37.55528",
    "lang": 0,
    "format": "json",
    "count": 1,
    "searchDttm": "202301011200"
}
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "appKey": "S0K9NJfpnp8Fh49tqvMyh3pagQWchEs83tkRAAaD"
}

response = requests.post(url, json=payload, headers=headers)

print(response.text)