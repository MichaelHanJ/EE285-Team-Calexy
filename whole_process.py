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

def whole_process(imagename):
	image_in = clipping_image(imagename) # return txt. file of paths of cropped images




if __name__ == "__main__":
	imagename = sys.argv[2] ## using python clip_image.py clip_image test_2.JPEG 
	whole_process(imagename)
