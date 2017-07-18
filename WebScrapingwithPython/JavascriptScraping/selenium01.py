from selenium import webdriver
import time

driver = webdriver.PhantomJS(executable_path='D:/02.Python/vEnvDJango_3_5_0_32/phantomjs-2.1.1-windows/bin/phantomjs')
driver.get('http://pythonscraping.com/pages/javascript/ajaxDemo.html')
time.sleep(3)
print(driver.find_element_by_id('content').text)
driver.close()