import requests

###################################################################################################
## Text POST 요청
###################################################################################################
params = {'firstname': 'Ryan', 'lastname': 'Mitchell'}
r = requests.post('http://pythonscraping.com/files/processing.php', data=params)
print(r.text)