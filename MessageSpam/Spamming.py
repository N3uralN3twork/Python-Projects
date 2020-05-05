"""Spam Script Sender - Word by Word
Author: Matthias Quinn
Source: https://github.com/HenryAlbu/FB-Messenger-Whatsapp-Discord-message-spammer
Date: 4th May 2020
Goals: Spam someone's inbox with a random text"""


from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import time
import platform


# Variables
sendDelay = 1
email = "miq_qedquinn@yahoo.com"
password = "cocoa1286"
friendName = "Matt Quinn [N3uralN3twork]"

# Checks if on Mac or Windows
if platform.system() == "Windows":
    driver = webdriver.Chrome('chromedriver.exe')
else:
    driver = webdriver.Chrome()

# Opens Discord
driver.get('https://discordapp.com/login')

# Login
driver.find_element_by_xpath('//*[@name="email"]').send_keys(email)
driver.find_element_by_xpath('//*[@name="password"]').send_keys(password)
driver.find_element_by_xpath('//*[@type="submit"]').click()

# Waits 8 seconds to finish loading page
time.sleep(8)

# Finds user in DM list
getUser = driver.find_element_by_xpath("//*[contains(text(), '" + friendName + "')]").click()

movie_script = []
with open('ShrekText.txt', "r") as f:
    for line in f.readlines():
        for word in line.split():
            print(word)
            # Types words and submits
            actions = ActionChains(driver)
            actions.send_keys(word, Keys.ENTER)
            actions.perform()
            time.sleep(sendDelay)
