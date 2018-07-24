from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import string
import operator

## string.digits      : 숫자
## string.hexdigits   : 16진수
## string.letters     : 영어 대문자 & 소문자
## string.lowercase   : 영어 소문자
## string.uppercase   : 영어 대문자
## string.punctuation : 특수문자
## string.printable   : 인쇄 가능한 모든 문자들
## string.whitespace  : 공백 문자 모두

def cleanInput(input):
    input = re.sub('\n+', ' ', input).lower()
    input = re.sub('\[[0-9]*\]', '', input)
    input = re.sub(' +', ' ', input)
    input = bytes(input, 'utf-8')
    input = input.decode('ascii', 'ignore')
    cleanInput = []
    input = input.split(' ')
    for item in input:
        item = item.strip(string.punctuation)
        if len(item) > 1 or (item.lower() == 'a' or item.lower() == 'i'):
            cleanInput.append(item)
    return cleanInput

def ngrams(input, n):
    input = cleanInput(input)
    output = {}
    for i in range(len(input)-n+1):
        ngramTemp = ' '.join(input[i:i+n])
        if ngramTemp not in output:
            output[ngramTemp] = 0
        output[ngramTemp] += 1
    return output

# 특정 단어 제거하기
def isCommon(ngrams):
    commonWords = ['the', 'be', 'and', 'of', 'a', 'in', 'to', 'have', 'it', 'i', 'that', 'is', 'an', 'at', 'but', 'we',
                   'his', 'from', 'not', 'by', 'she', 'or', 'as', 'what', 'go', 'their', 'can', 'who', 'get', 'if', 'would',
                   'her', 'all', 'my', 'make', 'about', 'know', 'will', 'as', 'up', 'one', 'time', 'has', 'been', 'there',
                   'year', 'so', 'think', 'when', 'which', 'them', 'some', 'me', 'people', 'take', 'out', 'inot', 'just',
                   'see', 'him', 'your', 'come', 'could', 'now', 'than', 'like', 'other', 'how', 'then', 'its', 'our',
                   'two', 'more', 'these', 'want', 'way', 'look', 'first', 'also', 'new', 'because', 'day', 'more', 'use',
                   'no', 'man', 'find', 'here', 'thing', 'give', 'many', 'well']

    i = len(ngrams)

    for word in ngrams[-1::-1]:
        for w in word[0].split(' '):
            if w in commonWords:
                ngrams.pop(i-1)
                break
        i -= 1
    return ngrams

content = str(urlopen('http://pythonscraping.com/files/inaugurationSpeech.txt').read(), 'utf-8')
ngrams = ngrams(content, 2)
sortedNGrams = sorted(ngrams.items(), key=operator.itemgetter(1), reverse=True)
sortedNGrams = isCommon(sortedNGrams)
print(sortedNGrams)
