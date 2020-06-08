# Projet_Classification_ZARA

Le dataset ZARA_Dataset_models  est un object de type dictionnaire. 

Il a été crée avec le jupyter notebook Image_treatment_and_dataset_creation.ipynb

    =================   =====================================
    Classes              4 : jupe, pantalon, robe ou t-shirt
    Samples total                                       1307
    =================   =====================================

#
    
Il contient les attributs suivants :
  
  - dataset.data : numpy array de taille (1307, 3136)
    chaque ligne correspond  à une image  traitée (resize, grayscale) de vetement "applatie". 
    ZARA_Dataset_models.data
  
 -  dataset.images : numpy array of shape (1307, 56, 56)
        Chaque ligne correspond  à une image de l'un des 1307 vêtements contenus dans le dataset.
        
 -  dataset.data_Name : numpy array of shape (1307,)
        Noms des images, ie (jupe_1, jupe_2, etc)
       
 - dataset.target : numpy array of shape (1307,)
        Labels sous forme encodée (ie. 0,1,2 ou 3) associés à chaque image de vétement. 

 - dataset.target_names : numpy array of shape (1307,)
        Labels sous forme catégorielle (ie. jupe, pantalon, robe ou t-shirt) associés à chaque image de vétement.
      
 - dataset.target_name_code : dictionnaire
        Correspondance label code <-> classes ie. {0: 'jupe', 1: 'pantalon', 2: 'robe', 3: 't-shirt'}
       
 - dataset.target_name_list :  liste des 4 classes
        [jupe','pantalon','robe','t-shirt']
 
 - dataset.hog_features : numpy array of shape (1307, 10368)
        Chaque ligne correspond au vecteur des histogrammes des gradients orientés (HOG).
 
 - dataset.hog_images : numpy array of shape (1307, 200,200)
        Chaque ligne correspond à une image traitée avec les histogrammes des gradients orientés (HOG).
 
#

Pour charger le dataset dans un script python :


```python

 import pickle

with open('dataset_path', 'rb') as data:
    dataset = pickle.load(data)  
```
 
