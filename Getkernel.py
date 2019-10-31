# --------------------------------------------------------------------------------------------------
# EE569 Homework Assignment #1
# Date: January 22, 2019
# Name: Suchismita Sahu
# ID: 7688176370
# email: suchisms@usc.edu
# --------------------------------------------------------------------------------------------------

from tensorflow.python.platform import flags
import pickle
import data
import saab

flags.DEFINE_string("output_path", None, "The output dir to save params")
flags.DEFINE_string("use_classes", "0-9", "Supported format: 0,1,5-9")
flags.DEFINE_string("kernel_sizes", "4,4",
                    "Kernels size for each stage. Format: '3,3'")
flags.DEFINE_integer(
    "stride", "4", "number of steps taken to move the kernel across image. Default: 1")
flags.DEFINE_string("num_kernels", "12,40",
                    "Num of kernels for each stage. Format: '4,10'")
flags.DEFINE_float("energy_percent", None,
                   "Energy to be preserved in each stage")
flags.DEFINE_integer("use_num_images", 10000,
                     "Num of images used for training")
FLAGS = flags.FLAGS


def main():
        # read data
    train_images, train_labels, test_images, test_labels, class_list = data.import_data(
        FLAGS.use_classes)
    print('Training image size:', train_images.shape)
    print('Testing_image size:', test_images.shape)

    kernel_sizes = saab.parse_list_string(FLAGS.kernel_sizes)
    stride = FLAGS.stride
    if FLAGS.num_kernels:
        num_kernels = saab.parse_list_string(FLAGS.num_kernels)
    else:
        num_kernels = None
    energy_percent = FLAGS.energy_percent
    use_num_images = FLAGS.use_num_images
    print('Parameters:')
    print('use_classes:', class_list)
    print('Kernel_sizes:', kernel_sizes)
    print('Stride:', stride)
    print('Number_kernels:', num_kernels)
    print('Energy_percent:', energy_percent)
    print('Number_use_images:', use_num_images)

    pca_params = saab.multi_Saab_transform(train_images, train_labels,
                                           kernel_sizes=kernel_sizes,
                                           stride=stride,
                                           num_kernels=num_kernels,
                                           energy_percent=energy_percent,
                                           use_num_images=use_num_images,
                                           use_classes=class_list)

    # save data
    fw = open('pca_params.pkl', 'wb')
    pickle.dump(pca_params, fw)
    fw.close()


if __name__ == '__main__':
    main()
