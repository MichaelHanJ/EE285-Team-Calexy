#from __future__ import division
#from Tkinter import *
#import tkMessageBox
from PIL import Image, ImageTk
import os
import glob
import random
import sys
import re
import numpy as np
#from numpy import linalg as LA
import pandas as pd
#%matplotlib inline 
import matplotlib.pyplot as plt 
#from time import gmtime, strftime
#from scipy.signal import butter, lfilter,savgol_filter
#from_future_import division
#import scipy as sp

def clipping_image(imagename):
    
    def dataseparation(txtlocation):
    
    #this function change the txt.file into list of tuples
    
    #input: fname: txt file location
    
    #output: list
   
    #title_akas = pd.read_csv('/Users/seankamano/Downloads/title.akas.tsv', delimiter = '\t', encoding = 'utf-8')
        f_1 = open(txtlocation, 'r') # read files
        lines = f_1.readlines() # read line by line
        f_1.close() # close the file
        new_lines = []
        for l in lines[:]:
            new_lines.append(l.split('\n')) # remove "\t" by line
    
        mHealth = pd.DataFrame(new_lines)
        loc_total = mHealth[0]
        num_item = loc_total.shape[0]
        loc_result = []
        for idx in range(num_item):
            loc_item = tuple(map(int, loc_total[idx].split(' ')))
            loc_result.append(loc_item)
        return loc_result
     
    def clipimage(imagelocation, savelocation, loc_result, imagename):
    #input: loc_result: a list of tuples which contain the location for each face
              #imagelocation: image location
              #saveflocation: location to store clipped images
              #imagename
       #output: clipped faces stored in the destination
        pathdir = os.getcwd()
        txt_cropped_image_path = pathdir + "/cropped_images/cropped_image_path.txt" # path of the saved paths of cropped images
        f = open(txt_cropped_image_path, "w")
        finalsavepath = pathdir + "/" + savelocation + "/" + "face_images"
        f.writelines(finalsavepath)
        f.close()
        files = glob.glob(finalsavepath + "/*")
        for d in files:
            os.remove(d) # remove all old images
        im = Image.open(imagelocation)
        for idx, loc in enumerate(loc_result):
            region = im.crop(loc)
            #region = region.resize((224, 224)) # resize the cropped image to 224x224
            savepath = finalsavepath + "/" + imagename[:-5] + "_cropped_" + str(idx) + ".JPEG"
            #f.writelines(savepath + "\n")
            region.save(savepath)
        #f.close()   
        return txt_cropped_image_path
        # quality and be changed or removed
     
    
    image_path = ""
    #with open('cropped_images/testexample/' + imagename[:-4] + '.txt') as f:
    with open('cropped_images/image_path.txt') as f:
        image_path = f.readlines()
    f.close()

    imagelocation = image_path[0]
    txtlocation = "locations/location.txt"
    saveloction = "cropped_images"
    imagename = imagename
    #print imagename
    loc_result = dataseparation(txtlocation)
    return clipimage(imagelocation, saveloction, loc_result, imagename)
    

if __name__ == "__main__":
    #function = getattr(sys.modules[__name__], sys.argv[1])
    imagename = sys.argv[1]
    clipping_image(imagename)
