from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np
import os, sys

path = "Raw_Images/"
dirs = os.listdir( path )
#color_128 = (128,128)



def resizing_reducing():
    
    #un boucle pour parcourir le dossier
    for item in dirs:
        if os.path.isfile(path+item):
            #item = 1 fichier du dossier
            
            #on ouvre le fichier/l'image, on la redimensionne, on la range dans un array
            #im = np.array(Image.open(path+item).convert('L').resize((28, 28)))
            im=Image.open(path+item).convert('L').resize((28, 28))
            
            
            #on floute l'image
            im.filter(ImageFilter.GaussianBlur(5))
            
            #on recup le texte pour renommer les fichiers
            f, e = os.path.splitext(path+item)
            
            #reduction de couleurs :
            #im = im // 128 * 128
            
            #ça ne marche toujours pas
            #ImageOps.invert(Image.fromarray(im))
            ImageOps.invert(im)
            
            #on augmente le contraste 
            #factor = 1.5 #increase contrast
            #ImageEnhance.Contrast(Image.fromarray(im)).enhance(factor)
            
            #on donne un nom au fichier généré à chaque tour de boucle
            im.save("treated/" + item  + '_blur5_inverted.jpg', 'JPEG', quality=90)
            
resizing_reducing()
