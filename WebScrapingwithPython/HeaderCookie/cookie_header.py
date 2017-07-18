from selenium import webdriver

phantomJS_path = 'D:\\KYH\\02.PYTHON\\vEnvDjango3_5_2_64\\phantomjs-2.1.1-windows\\bin\\phantomjs'

driver = webdriver.PhantomJS(executable_path=phantomJS_path)
driver.get('http://pythonscraping.com')
driver.implicitly_wait(1)
print(driver.get_cookies())

savedCookies = driver.get_cookies()
driver2 = webdriver.PhantomJS(executable_path=phantomJS_path)
driver2.get('http://pythonscraping.com')
driver2.delete_all_cookies()

for cookie in savedCookies:
    driver2.add_cookie(cookie)  # 기존 cookie 를 재사용

driver2.get('http//pythonscraping.com')
driver2.implicitly_wait(1)
print(driver2.get_cookies())