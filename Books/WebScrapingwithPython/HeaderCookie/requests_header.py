import requests
from bs4 import BeautifulSoup

session = requests.Session()
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit 537.36 (KHTML, like Gecko) Chrome',
           'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'}
url = 'https://www.whatismybrowser.com/developers/what-http-headers-is-my-browser-sending'  # URL 이 잘못됨
req = session.get(url, headers=headers)
print(req.text)
bsObj = BeautifulSoup(req.text, 'html.parser')
print(bsObj.find('table', {'class': 'table-striped'}).get_text())