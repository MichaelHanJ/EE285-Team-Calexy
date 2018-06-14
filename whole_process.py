from PIL import Image, ImageTk
import os
import glob
import random
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from clip_image import clipping_image
import classifier

def whole_process(imagename):
	image_in = clipping_image(imagename) # return path of txt. file which contains the paths of cropped images
	
	# read path
	with open(image_in,'r') as f:
		im_path = f.read()
	
	classifier.classify(im_path) # print all the predictions for each image under im_path



if __name__ == "__main__":
	imagename = sys.argv[1] ## using python whole_process.py test_2.JPEG 
	whole_process(imagename)
