
# --------------------------------------------------------------------------------------------------
# EE569 Homework Assignment #6
# Date: April 28, 2019
# Name: Suchismita Sahu
# ID: 7688176370
# email: suchisms@usc.edu
# --------------------------------------------------------------------------------------------------

import pickle
from keras.datasets import mnist
import numpy as np
from saab import *
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # load pca params
    fr = open('pca_params.pkl', 'rb')
    pca_params = pickle.load(fr)
    fr.close()

    # read test images
    test_images = []
    for i in range(4):
        img = plt.imread(str(i+1)+'.png')
        test_images.append(img)

    test_images = np.vstack(test_images)
    test_images = test_images.reshape(-1, 32, 32, 1)
    test_images = test_images/255.

    print('Testing_image size:', test_images.shape)

    print('Getting Saab co-efficients...')

    output = initialize(test_images, pca_params)

    print('Reconstructing images...')
    
    data = output
    for i in range(1, -1, -1):

        kernel = pca_params[f'Layer_{i}/kernel']
        feature_mean = pca_params[f'Layer_{i}/feature_expectation']

        if i != 0:
            bias = pca_params[f'Layer_{i}/bias']
            prev_kernel = pca_params[f'Layer_{i-1}/kernel']
            e = np.zeros((1, kernel.shape[0]))
            e[0, 0] = 1
            data += bias*e
            data = np.matmul(data, np.linalg.pinv(np.transpose(kernel)))
            data -= np.sqrt(kernel.shape[1])*bias
            data = data.reshape(-1, data.shape[-1])
            data += feature_mean
            data = data.reshape(test_images.shape[0],2,2,1,1,4,4,prev_kernel.shape[0])
            data = data.transpose(0,1,3,5,2,4,6,7).reshape(test_images.shape[0],8,8,prev_kernel.shape[0])
        else:
            data = np.matmul(data, np.linalg.pinv(np.transpose(kernel)))
            data = data.reshape(-1, data.shape[-1])
            data += feature_mean
            data = data.reshape(test_images.shape[0],8,8,1,1,4,4,1)
            data = data.transpose(0,1,3,5,2,4,6,7).reshape(test_images.shape[0],32,32,1)

print(data.shape)
true = test_images[0].squeeze(-1)
recon = data[0].squeeze(-1)

s = 0
for i in range(true.shape[0]):
    for j in range(true.shape[1]):
        s += (true[i][j] - recon[i][j])**2
s = s / (true.shape[0] * true.shape[1])

max_val = np.max(true)
mse_sqrt = np.sqrt(s)
psnr = 20 * np.log10(max_val / mse_sqrt)

print(psnr)

plt.imshow(recon, cmap='gray')
plt.show()
