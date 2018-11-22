from scipy.misc import imread
import numpy as np
import pandas as pd
import os
root = './train' #DATA FILE WHICH WILL CONVERT TO.CSV
# IT WILL GO THROUGH EACH FOLDER IN ROOT FOLDER
for directory, subdirectories, files in os.walk(root):
    # go through each file in that directory
	for file in files:	
        # read the image file and extract its pixels
		im = imread(os.path.join(directory,file))
		value = im.flatten()
        # I renamed the folders containing digits to the contained digit itself. For example, digit_0 folder was renamed to 0.
        # so taking the 9th value of the folder gave the digit (i.e. "./train/8" ==> 9th value is 8), which was inserted into the first column of the dataset.
		value = np.hstack((directory[8:],value))
		df = pd.DataFrame(value).T
        df = df.sample(frac=1) # shuffle the dataset
		with open('train.csv', 'a') as dataset: 
			df.to_csv(dataset, header=False, index=False)
