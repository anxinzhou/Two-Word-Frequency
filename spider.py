from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from multiprocessing.pool import ThreadPool
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor

inifinit = 2 << 32


def get_tickit():
    driver = webdriver.Chrome(ChromeDriverManager().install())
    executor = ThreadPoolExecutor(max_workers=10)
    driver.get("http://msg.cityline.com/busy.html")
    driver.implicitly_wait(5)
    current_url = driver.current_url
    for i in range(inifinit):
        elem = driver.find_element_by_id("btn-retry-en-1")
        elem.click()
        if driver.current_url != current_url:
            break
        time.sleep(3)
    e = driver.find_element_by_id("buyBtWalkIn")
    e.click()
    time.sleep(1000000)


for i in range(20):
    p = multiprocessing.Process(target=get_tickit)
    p.start()
# print(elems[0])
# boxList = []
# for e in elems:
#     boxList.append(e.find_element_by_tag_name("a"))
#
# elems = driver.find_elements_by_class_name("subject")
# messageList = []
# for e in elems:
#     messageList.append(e.find_element_by_tag_name("a"))
#
# e = messageList[0]
# e.click()
# print(driver.find_element_by_tag_name("pre").text)
# driver.execute_script("window.history.go(-1)")
