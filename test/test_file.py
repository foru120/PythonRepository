import numpy as np

a = np.array([ [[1, 2, 3],
                [2, 1, 4],
                [5, 2, 1],
                [6, 3, 2]],
               [[5, 1, 3],
                [1, 3, 4],
                [4, 2, 6],
                [3, 9, 3]],
               [[4, 5, 6],
                [7, 4, 3],
                [2, 1, 5],
                [4, 3, 1]] ])

# print(np.argmax(a, axis=2))  # [1, 2, 3] -> 2, [2, 1, 4] -> 2, [5, 2, 1] -> 0, [6, 3, 2] -> 0 ::=> [[2 2 0 0], [0 2 2 1], [2 0 2 0]]
# print(np.argmax(a, axis=1))  # [1, 2, 5, 6] -> 3, [2, 1, 2, 3] -> 3, [3, 4, 1, 2] -> 1 ::=> [[3 3 1], [0 3 2], [1 0 0]]
# print(np.argmax(a, axis=0))  # [1, 5, 4] -> 1, [2, 1, 5] -> 2, [3, 3, 6] -> 2 ::=> [[1 2 2], [2 2 0], [0 0 1], [0 1 1]]

b = np.array([1, 3, 4, 2])
c = np.array([5, 3, 2, 1])
print(np.mean(b == c))

v = [1, 2, 3, 4, 5]
print(np.mean(v))

from lxml import html
import requests
import re
import json
import urllib
import sys

path = 'D:\\data\\image\\'


domain = 'https://openi.nlm.nih.gov/'
url_list = []
for i in range(0,75):
    url = 'https://openi.nlm.nih.gov/gridquery.php?q=&it=x,xg&sub=x&m='+str(1+100*i)+'&n='+str(100+100*i)
    url_list.append(url)
regex = re.compile(r"var oi = (.*);")
final_data = {}
img_no = 0


def extract(url):
    global img_no

    img_no += 1
    r = requests.get(url)
    tree = html.fromstring(r.text)

    div = tree.xpath('//table[@class="masterresultstable"]\
        //div[@class="meshtext-wrapper-left"]')

    if div != []:
        div = div[0]
    else:
        return

    typ = div.xpath('.//strong/text()')[0]
    items = div.xpath('.//li/text()')
    img = tree.xpath('//img[@id="theImage"]/@src')[0]


    final_data[img_no] = {}
    final_data[img_no]['type'] = typ
    final_data[img_no]['items'] = items
    final_data[img_no]['img'] = domain + img
    urllib.request.urlretrieve(domain+img, path+str(img_no)+".png")
    with open('data_new.json', 'w') as f:
        json.dump(final_data, f)
    print(final_data[img_no])


def main():
    for url in url_list :
        r = requests.get(url)
        tree = html.fromstring(r.text)

        script = tree.xpath('//script[@language="javascript"]/text()')[0]

        json_string = regex.findall(script)[0]
        json_data = json.loads(json_string)

        next_page_url = tree.xpath('//footer/a/@href')

        print('extract')
        links = [domain + x['nodeRef'] for x in json_data]
        for link in links:
            extract(link)

if __name__ == '__main__':

    main()


#python scraper.py <path to folders>