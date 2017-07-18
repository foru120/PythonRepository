from urllib.request import urlretrieve
from urllib.request import urlopen
from bs4 import BeautifulSoup
import subprocess
import requests
from PIL import Image, ImageOps

def cleanImage(imagePath):
    image = Image.open(imagePath)
    image = image.point(lambda x: 0 if x < 143 else 255)
    borderImage = ImageOps.expand(image, border=20, fill='white')
    borderImage.save(imagePath)

html = urlopen('http://www.pythonscraping.com/humans-only')
bsObj = BeautifulSoup(html.read(), 'html.parser')

# 미리 만들어진 폼 값을 수집합니다.
imageLocation = bsObj.find('img', {'title': 'Image CAPTCHA'})['src']
formBuildId = bsObj.find('input', {'name': 'form_build_id'})['value']  # hidden parameter
captchaSid = bsObj.find('input', {'name': 'captcha_sid'})['value']
captchaToken = bsObj.find('input', {'name': 'captcha_token'})['value']
captchaUrl = 'http://pythonscraping.com'+imageLocation
urlretrieve(captchaUrl, 'ScrapedImages/captcha.jpg')
cleanImage('ScrapedImages/captcha.jpg')
p = subprocess.Popen(['tesseract', 'ScrapedImages/captcha.jpg', 'ScrapedImages/captcha'],
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
p.wait()
f = open('ScrapedImages/captcha.txt', 'r', encoding='utf-8')

# 공백을 제거합니다.
captchaResponse = f.read().replace(' ', '').replace('\n', '')
print('Captcha solution attempt : ' + captchaResponse)

if len(captchaResponse) == 5:
    params = {'captcha_token': captchaToken, 'captcha_sid': captchaSid, 'form_id': 'comment_node_page_form',
              'form_build_id': formBuildId, 'captcha_response': captchaResponse, 'name': 'Ryan Mitchell',
              'subject': 'I come to seek the Grail', 'comment_body[und][0][value]': '...and I am definitely not a bot'}
    r = requests.post('http://www.pythonscraping.com/comment/reply/10', data=params)
    responseObj = BeautifulSoup(r.text, 'html.parser')
    if responseObj.find('div', {'class': 'messages'}) is not None:
        print(responseObj.find('div', {'class': 'messages'}).get_text())
    else:
        print('There was a problem reading the CAPTCHA correctly!')