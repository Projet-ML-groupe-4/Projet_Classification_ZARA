import requests
import io
import os
from PIL import Image
import hashlib

import csv
import time
import random
import sys
 
# selenium package
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import UnexpectedAlertPresentException
from selenium.common.exceptions import WebDriverException

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import Select
import multiprocessing
from threading import Thread
from time import sleep
import multiprocessing

# telechargement des images

def get_image(folder_path:str,url:str):
    try:
        image_content = requests.get(url).content

    except Exception as e:
        print(f"ERROR - Could not download {url} - {e}")

    try:
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert('RGB')
        file_path = os.path.join(folder_path,hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')
        with open(file_path, 'wb') as f:
            image.save(f, "JPEG", quality=85)
        print(f"SUCCESS - saved {url} - as {file_path}")
    except Exception as e:
        print(f"ERROR - Could not save {url} - {e}")

# Les urls des chaques categories

urls_ = ['https://www.zara.com/fr/fr/femme-robes-l1066.html?v1=1445722',
         'https://www.zara.com/fr/fr/femme-jupes-l1299.html?v1=1445719',
        'https://www.zara.com/fr/fr/femme-t-shirts-l1362.html?v1=1445717',
        'https://www.zara.com/fr/fr/femme-pantalons-l1335.html?v1=1445724']


# les paths pour enregistre les images

file_zalando = (r"\Users\utilisateur\Desktop\projet_ML\Projet_Classification_ZARA\zara\news")
file_robe = (r"\Users\utilisateur\Desktop\projet_ML\Projet_Classification_ZARA\zara\robes")
file_jupe = (r"\Users\utilisateur\Desktop\projet_ML\Projet_Classification_ZARA\zara\jupes")
file_tshi = (r"\Users\utilisateur\Desktop\projet_ML\Projet_Classification_ZARA\zara\tshirts")
file_pant = (r"\Users\utilisateur\Desktop\projet_ML\Projet_Classification_ZARA\zara\pantalons")


# configurer webdriver

capa = DesiredCapabilities.CHROME
capa["pageLoadStrategy"] = "none"
driver_1 = webdriver.Chrome(desired_capabilities=capa)
driver_2 = webdriver.Chrome(desired_capabilities=capa)
driver_3 = webdriver.Chrome(desired_capabilities=capa)
driver_4 = webdriver.Chrome(desired_capabilities=capa)
wait1 = WebDriverWait(driver_1, 20)
wait2 = WebDriverWait(driver_2, 20)
wait3 = WebDriverWait(driver_3, 20)
wait4 = WebDriverWait(driver_1, 20)
driver_1.get(urls_[0])
driver_2.get(urls_[1])
driver_3.get(urls_[2])
driver_4.get(urls_[3])


# Fonction pour scrapper les Pantalon

def pantalons_s():
    for i in range(459):
        try:
            liste_products_s = driver_4.find_elements_by_class_name('product')
            print(f'nombres des robes {int(len(liste_products_s))-1}')

            # click sur image
            liste_products_s[i].click()
            time.sleep(4)
            # get images
            src_img = driver_4.find_element_by_css_selector("#main-images > div:nth-child(1) > a > img.image-big._img-zoom._main-image").get_attribute('src')
            get_image(file_pant,src_img )
            print(f'pantalon numero {i} enregistré')
            sleep(5)
            driver_4.back()
            sleep(2)
        except(WebDriverException):
                print("lien ou chemin introuvable")


# Fonction pour scrapper les Robes

def robes_s():
    for i in range(85, 652):
        try:
            liste_products_s = driver_1.find_elements_by_class_name('product')
            print(f'nombres des robes restantes {int(len(liste_products_s))-i}')
            # click sur image
            liste_products_s[i].click()
            time.sleep(4)
            # get images
            src_img = driver_1.find_element_by_css_selector("#main-images > div:nth-child(1) > a > img").get_attribute('src')
            url_images.append(src_img)
            get_image(file_robe,src_img )
            print(f'robe numero : {i} enregistrée')
            driver_1.back()
            time.sleep(3)
        except(WebDriverException):
                print("lien ou chemin introuvable")

# Fonction pour scrapper les jupes_s

def jupes_s():
    for i in range(3):
        try:
            liste_products_s = driver_2.find_elements_by_class_name('product')
            #print(list_company_names[i].text)

            # click sur image
            liste_products_s[i].click()
            time.sleep(3)
            # get images
            src_img = driver_2.find_element_by_css_selector("#main-images > div:nth-child(1) > a > img").get_attribute('src')
            url_images.append(src_img)
            print(f'src {i} enregistre')
            sleep(5)
            driver_2.back()
            get_image(file_jupe,src_img )
            time.sleep(3)
        except(WebDriverException):
                print("lien ou chemin introuvable")

# Fonction pour scrapper les tshirt

def tshirts_s():
    for i in range(70, 803):
        liste_products_s = driver_3.find_elements_by_class_name('product')
        try:
            produits = liste_products_s
            print(f'nombres des robes {int(len(liste_products_s))-i}')

            # click sur image
            produits[i].click()
            time.sleep(4)
            # get images
            src_img = driver_3.find_element_by_css_selector("#main-images > div:nth-child(1) > a > img.image-big._img-zoom._main-image").get_attribute('src')
            url_images.append(src_img)
            get_image(file_tshi,src_img )
            print(f'tshirt numero {i} enregistré')
            sleep(2)
            driver_3.back()
            time.sleep(2)
        except(WebDriverException):
                print("lien ou chemin introuvable")


# multiprocessing

start = time.perf_counter()
pantalons_s()
robes_s()

p1 = multiprocessing.process(target=pantalons_s)
p2 = multiprocessing.process(target=robes_s)
p3 = multiprocessing.process(target=jupes_s)
p4 = multiprocessing.process(target=tshirts_s)

p1.join()
p2.join()
p3.join()
p4.join()

finish = time.perf_counter()
print(f'Finished in {round(finish - start, 2)} secondes')