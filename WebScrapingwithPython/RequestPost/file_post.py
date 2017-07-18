import requests

###################################################################################################
## File POST 요청
###################################################################################################
files = {'uploadFIle': open('../files/Python-logo.png', 'rb')}
r = requests.post('http://pythonscraping.com/pages/processing2.php', files=files)
print(r.text)