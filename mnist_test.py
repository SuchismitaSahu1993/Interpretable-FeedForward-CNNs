# --------------------------------------------------------------------------------------------------
# EE569 Homework Assignment #6
# Date: April 28, 2019
# Name: Suchismita Sahu
# ID: 7688176370
# email: suchisms@usc.edu
# --------------------------------------------------------------------------------------------------

import data
import saab
import pickle
import numpy as np
import sklearn
import cv2
import keras
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

def main():

	# load data
	fr=open('pca_params_E5S5.pkl','rb')  
	pca_params=pickle.load(fr, encoding='latin1')
	fr.close()

	# read data
	train_images, train_labels, test_images, test_labels, class_list = data.import_data("0-9")
	print('Testing_image size:', test_images.shape)

	# testing
	print('--------Testing--------')
	feature=saab.initialize(test_images, pca_params)
	feature=feature.reshape(feature.shape[0],-1)
	print("S4 shape:", feature.shape)
	print('--------Finish Feature Extraction subnet--------')

	# feature normalization
	# std_var=(np.std(feature, axis=0)).reshape(1,-1)
	# feature=feature/std_var
	
	num_clusters=[120, 80, 10]
	use_classes=10

	fr=open('llsr_weights_E5S5.pkl','rb')  
	weights=pickle.load(fr,encoding='latin1')
	fr.close()

	fr=open('llsr_bias_E5S5.pkl','rb')  
	biases=pickle.load(fr,encoding='latin1')
	fr.close()

	for k in range(len(num_clusters)):

		# least square regression
		weight=weights['%d LLSR weight'%k]
		bias=biases['%d LLSR bias'%k]
		feature=np.matmul(feature,weight)+bias
		print(k,' layer LSR weight shape:', weight.shape)
		print(k,' layer LSR bias shape:', bias.shape)
		print(k,' layer LSR output shape:', feature.shape)
		
		if k!=len(num_clusters)-1:
			# Relu
			for i in range(feature.shape[0]):
				for j in range(feature.shape[1]):
					if feature[i,j]<0:
						feature[i,j]=0
		else:
			pred_labels=np.argmax(feature, axis=1)
			acc_test=sklearn.metrics.accuracy_score(test_labels,pred_labels)
			print('testing acc is {}'.format(acc_test))


	fw=open('test_pred_E5S5.pkl','wb')    
	pickle.dump(feature, fw)    
	fw.close()


if __name__ == '__main__':
	main()
