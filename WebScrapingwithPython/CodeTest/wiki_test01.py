import unittest
from bs4 import BeautifulSoup
from urllib.request import urlopen, unquote

bsObj = None
url = None

class TestWikipedia(unittest.TestCase):

    def test_PageProperties(self):
        global bsObj
        global url

        url = 'http://en.wikipedia.org/wiki/Monty_Python'

        # 처음 100 페이지를 테스트합니다.
        for i in range(1, 100):
            bsObj = BeautifulSoup(urlopen(url), 'html.parser')
            titles = self.titleMatchesURL()
            self.assertEquals(titles[0], titles[1])
            self.assertTrue(self.contentExists())

        print('Done!')

    def titleMatchesURL(self):
        global bsObj
        global url
        pageTitle = bsObj.find('h1').get_text()
        urlTitle = url[(url.index('/wiki/')+6):]
        urlTitle = urlTitle.replace('_', ' ')
        urlTitle = unquote(urlTitle)
        return [pageTitle.lower(), urlTitle.lower()]

    def contentExists(self):
        global bsObj
        content = bsObj.find('div', {'id': 'mw-content-text'})
        if content is not None:
            return True
        return False

if __name__=='__main__':
    unittest.main()
