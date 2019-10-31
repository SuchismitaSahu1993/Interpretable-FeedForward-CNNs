
# --------------------------------------------------------------------------------------------------
# EE569 Homework Assignment #6
# Date: April 28, 2019
# Name: Suchismita Sahu
# ID: 7688176370
# email: suchisms@usc.edu
# --------------------------------------------------------------------------------------------------

import pickle
import numpy as np
import data
import os
import sklearn
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

files = []
for (dirpath, dirnames, filename) in os.walk(os.curdir):
    files.extend(filename)

train_files = []
test_files = []
for f in files:
    if 'train' in f and 'pkl' in f.split('.'):
        train_files.append(f)
    if 'test' in f and 'pkl' in f.split('.'):
        test_files.append(f)

train_pred = []
test_pred = []
for file in train_files:
    fr = open(file, 'rb')
    train_pred.append(pickle.load(fr))
    fr.close()

for file in test_files:
    fr = open(file, 'rb')
    test_pred.append(pickle.load(fr))
    fr.close()

train_pred = np.concatenate(train_pred, axis=1)
test_pred = np.concatenate(test_pred, axis=1)

pca = PCA(n_components=10, svd_solver='full')
x_train = pca.fit_transform(train_pred)

x_test = pca.transform(test_pred)

train_images, train_labels, test_images, test_labels, class_list = data.import_data("0-9")

clf = SVC(gamma='auto')
clf.fit(x_train, train_labels)

res = clf.predict(x_train)

acc_test = sklearn.metrics.accuracy_score(train_labels, res)
print(acc_test)

cm = confusion_matrix(y_target=test_labels, 
                  y_predicted=res, 
                  binary=False)

fig, ax = plot_confusion_matrix(conf_mat=cm)
plt.savefig('ffcnn_cm.png')
plt.show()
