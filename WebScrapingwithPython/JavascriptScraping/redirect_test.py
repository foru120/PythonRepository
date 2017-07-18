from selenium import webdriver
import time
from selenium.webdriver.remote.webelement import WebElement
from selenium.common.exceptions import NoSuchElementException

def waitForLoad(driver):
    elem = driver.find_element_by_tag_name('html')
    count = 0

    while True:
        count += 1
        if count > 20:
            print('Timing out after 10 seconds and returning')
            return
        time.sleep(.5)

        try:
            elem = driver.find_element_by_tag_name('div')
        except NoSuchElementException:
            return

driver = webdriver.PhantomJS(executable_path='D:/02.Python/vEnvDJango_3_5_0_32/phantomjs-2.1.1-windows/bin/phantomjs')
driver.get('http://pythonscraping.com/pages/javascript/redirectDemo1.html')
waitForLoad(driver)
print(driver.page_source)