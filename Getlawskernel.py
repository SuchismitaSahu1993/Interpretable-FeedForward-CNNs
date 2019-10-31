# --------------------------------------------------------------------------------------------------
# EE569 Homework Assignment #6
# Date: April 28, 2019
# Name: Suchismita Sahu
# ID: 7688176370
# email: suchisms@usc.edu
# --------------------------------------------------------------------------------------------------

from tensorflow.python.platform import flags
import pickle
import data
import saab
import numpy as np

flags.DEFINE_string("output_path", None, "The output dir to save params")
flags.DEFINE_string("use_classes", "0-9", "Supported format: 0,1,5-9")
flags.DEFINE_string("kernel_sizes", "5,5", "Kernels size for each stage. Format: '3,3'")
flags.DEFINE_string("num_kernels", "5,15", "Num of kernels for each stage. Format: '4,10'")
flags.DEFINE_float("energy_percent", None, "Energy to be preserved in each stage")
flags.DEFINE_integer("use_num_images", -1, "Num of images used for training")
FLAGS = flags.FLAGS

#filter convolution
def convolve(i,j,filt,img):
    val = 0
    for m in range(-2,3):
        for n in range(-2,3):
            val += img[i+m][j+n]*filt[m+2][n+2]
    return val

def main():
    # read data
    train_images, train_labels, test_images, test_labels, class_list = data.import_data(FLAGS.use_classes)
    print('Training image size:', train_images.shape)
    print('Testing_image size:', test_images.shape)

        #Laws Filter creation
    L5 = np.array([1,4,6,4,1]).reshape(5,1)
    E5 = np.array([-1,-2,0,2,1]).reshape(5,1)
    S5 = np.array([-1,0,2,0,-1]).reshape(5,1)
    R5 = np.array([-1,2,0,-2,1]).reshape(5,1)
    W5 = np.array([1,-4,6,-4,1]).reshape(5,1)


    laws_filters = {'L5':L5,'E5':E5,'S5':S5,'R5':R5,'W5':W5}

    _2d_laws_filters = {}
    for k1,v1 in laws_filters.items():
        for k2,v2 in laws_filters.items():
            _2d_laws_filters[k1+k2] = np.matmul(v1,v2.T)


    #boundary extension by pixel replication
    extended_images  = []
    for img in train_images[:10000,:,:,0]:
        new_img = np.pad(img, 2,'reflect')
        extended_images.append(new_img) 

    #Laws feature extraction
    final_images = []
    for img in extended_images:
        new_img = np.empty((1,32,32), np.uint8)
        for i in range(2,32+2):
            for j in range(2,32+2):
                new_img[0][i-2][j-2] = convolve(i,j,_2d_laws_filters['S5R5'],img) 
        final_images.append(new_img)
    train_images = np.vstack(final_images)
    train_images = train_images.reshape(-1, 32, 32, 1)
    print(train_images.shape)
    

    kernel_sizes=saab.parse_list_string(FLAGS.kernel_sizes)
    if FLAGS.num_kernels:
        num_kernels=saab.parse_list_string(FLAGS.num_kernels)
    else:
        num_kernels=None
    energy_percent=FLAGS.energy_percent
    use_num_images=FLAGS.use_num_images
    print('Parameters:')
    print('use_classes:', class_list)
    print('Kernel_sizes:', kernel_sizes)
    print('Number_kernels:', num_kernels)
    print('Energy_percent:', energy_percent)
    print('Number_use_images:', use_num_images)

    pca_params=saab.multi_Saab_transform(train_images, train_labels,
                         kernel_sizes=kernel_sizes,
                         num_kernels=num_kernels,
                         energy_percent=energy_percent,
                         use_num_images=use_num_images,
                         use_classes=class_list)
                         
    #print(pca_params)                         
    # save data
    fw=open('pca_params_S5R5.pkl','wb')    
    pickle.dump(pca_params, fw)    
    fw.close()

if __name__ == '__main__':
    main()
