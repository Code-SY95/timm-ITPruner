import requests

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg"
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url, headers=headers)

print("✅ 응답 상태코드:", response.status_code)
print("✅ Content-Type:", response.headers.get("Content-Type"))
