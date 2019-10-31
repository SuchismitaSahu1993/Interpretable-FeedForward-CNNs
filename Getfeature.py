// --------------------------------------------------------------------------------------------------
// EE569 Homework Assignment #6
// Date: April 28, 2019
// Name: Suchismita Sahu
// ID: 7688176370
// email: suchisms@usc.edu
// --------------------------------------------------------------------------------------------------

import pickle
import numpy as np
import data
import saab
import keras
import sklearn

def main():
    
	# load data
	fr=open('pca_params_E5S5.pkl','rb')  
	pca_params=pickle.load(fr)
	fr.close()

	# read data
	train_images, train_labels, test_images, test_labels, class_list = data.import_data("0-9")
	print('Training image size:', train_images.shape)
	print('Testing_image size:', test_images.shape)
	
	batch_size = 100
	num_samples = int(len(train_images)/batch_size)
	
	# Training
	print('--------Training--------')
	features = []
	for i in range(num_samples):
		trn_images = train_images[i*batch_size:i*batch_size+batch_size,:]		
		feature=saab.initialize(trn_images, pca_params) 
		feature=feature.reshape(feature.shape[0],-1)
		features.append(feature)
	feature = np.vstack(features)
	print("S4 shape:", feature.shape)
	print('--------Finish Feature Extraction subnet--------')
	feat={}
	feat['feature']=feature
	
	# save data
	fw=open('feat_E5S5.pkl','wb')    
	pickle.dump(feat, fw)    
	fw.close()

if __name__ == '__main__':
	main()
