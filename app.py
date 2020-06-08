# -*- coding: utf-8 -*-
"""
A simple streamlit app to predict the type of clothe from a image
run the app by installing streamlit with pip and typing
> streamlit run clothing_classification.py

The Dataset  using for the model was created using images scraped from ZARA website. 
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.pipeline import Pipeline
# ML packages
from sklearn import mixture
from sklearn.mixture import GaussianMixture 
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from matplotlib import cm
from matplotlib.colors import LogNorm
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder,MinMaxScaler
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, precision_score, recall_score, r2_score, mean_absolute_error, mean_squared_error, mean_absolute_error
##SVM
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from mlxtend.preprocessing import shuffle_arrays_unison
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_confusion_matrix


import pickle
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from PIL import Image, ImageOps, ImageEnhance
import os, sys
import numpy as np
import mahotas
import cv2
import h5py
import hashlib # permet de chiffrer 
import time


# get the current directory 
DATADIR= os.getcwd()

#ZARA logo
#logo = Image.open('ZARA_logo.png').convert('RGB')
#st.image(logo, width=200)

# page setting
st.title("Clothing classification  :dress: :shirt: :jeans: ")
st.write("This application perform the prediction of the type of a clothe from a picture ")



def classifying():
    ## upload clothe image 
    uploaded_file = st.file_uploader("Upload your  image...",type=["png", "jpg", "jpeg"])

    if uploaded_file is None:
        img_array = None


    if uploaded_file is not None:
        img= Image.open(uploaded_file)
        st.image(img, width=200)
        #file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        #opencv_image = cv2.imdecode(file_bytes, 1)
        img_array = np.array(img)




    def traitement_image(img_array):
        fixed_size = tuple((56, 56))
        ### Load image ###
        #img = img_array
        ### resize ###
        resized=cv2.resize(img_array, fixed_size, interpolation=cv2.INTER_AREA)
        ### convert the image to grayscale ###
        imgGray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)  
        ### flatten
        imgGray_flat=np.array(imgGray).flatten()
        st.image(imgGray)
        return imgGray,imgGray_flat.reshape(1, -1)

    new_clothe_image , new_clothe= traitement_image(img_array)
    print(new_clothe.shape)

    # load Data 
    @st.cache(allow_output_mutation=True)


    def load_data():
        with open(DATADIR +'/'+ 'ZARA_DataSet_models.pickle', 'rb') as data:
            DataSet = pickle.load(data)  
        return DataSet

    DataSet = load_data()

#data_load_state = st.text('Loading data...')
#data_load_state.text("Done! (using st.cache)")

    for key in DataSet.keys():
        globals()[str(key)] =DataSet[key]

    # Train/test split 
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=0, stratify=target)
    class_names=target_name_list



##### Model ######

#model_name='SVM' # selection du model 

    model_selection=st.sidebar.selectbox("Please choose a classifier Algrithm", ("SVM", "Logistic Regression","Decision Tree", "Random Forest", "KNN", "Gaussian Bayesian", "AdaBoost", "GradBoost"))

    if model_selection == "SVM":
        model_name = 'SVM'
    elif model_selection == "Logistic Regression":
        model_name = 'LogR'
    elif model_selection == "Decision Tree":
        model_name = 'DecisionTree'
    elif model_selection =="Random Forest":
        model_name = 'RF'
    elif model_selection =="KNN":
        model_name = 'KNN'
    elif model_selection =="Gaussian Bayesian":
        model_name = "GaussianNB"
    elif model_selection =="AdaBoost":
        model_name = 'AdaBoost'
    elif model_selection =="GradBoost":
        model_name = "GradBoost"

    n_components=0.95 # nombre de composants pour la PCA
    #model={'SVM' : SVC(kernel= 'rbf',C=19.306977288832496, gamma=0.004094915062380427, random_state=0)}
    models={'SVM' : SVC(kernel= 'rbf',C=3.5564803062231287, gamma=0.022229964825261943, random_state=0),
             'GaussianNB':GaussianNB(),
             #'BernouilliNB':BernoulliNB(),
             'LogR': LogisticRegression(C=1.0, penalty= 'l2', random_state=0),
             'DecisionTree': DecisionTreeClassifier(max_depth=6, max_leaf_nodes=15, random_state=0,min_samples_split= 2),
             'RF': RandomForestClassifier(bootstrap=False, max_depth=20, random_state=0, max_features= 'auto', min_samples_leaf= 1, min_samples_split= 2, n_estimators= 100),
             'KNN':KNeighborsClassifier(metric='manhattan', n_neighbors=3, weights='distance'),
             'AdaBoost':AdaBoostClassifier(n_estimators=100, random_state=0),
             'GradBoost':GradientBoostingClassifier(learning_rate= 0.1, n_estimators=100,random_state=0)
            }


    def Modelisation_and_prediction(model_name:str, new_clothe):
        clf = Pipeline([
            ('scale', StandardScaler()),
            ('pca', PCA(n_components=n_components, whiten=True,  svd_solver='full')),
            (model_name, models[model_name])
                    ])

        clf.fit(X_train, y_train)
        class_pred = clf.predict(new_clothe)
        print(class_pred)
        print(class_names)
        return class_names[class_pred[0]]

    ## prediction de la classe du vetement présent sur l'image 
    new_clothe_class=Modelisation_and_prediction(model_name, new_clothe)
    st.write(f" It is a {new_clothe_class}")



classifying()
