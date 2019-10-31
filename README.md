## Interpretable-FeedForward-CNNs
Interpretable Feedforward Convolutional Neural Networks is a Novel Mothod developed by Prof. C.C Jay Kuo at University of Southern California, Department of Electrical Engineering. This network is implemented using Saab transform in cascade with Least Square Regressor for constructing the layers. These are more interpretable than traditional CNNs. Tested on MNIST Handwritten digit dataset.

## Requirements
1. Ubuntu/Windows
2. Python 3.6 + (preferably with anaconda)
3. PyTorch 1.0 +
4. Numpy, Pandas, matplotlib, scikit-learn
5. Keras

## General Usage Information
For running Image Reconstruction, first run Getkernel.py using appropriate filter settings. Also comment out maxpooling operation in saab.py
```
python Getkernel.py
```
Then run inverse_saab_transform.py to get reconstructed image. This will output single image and PSNR value. For different test images, we change index in true and recon variable as
```
true = test_images[1].squeeze(-1) # (index 1 represents 2nd image)
recon = data[1].squeeze(-1)

python inverse_saab_transform.py
```

For ensemble design, first run FF-CNN using different settings explained in the report. For changing settings with filter size, we run Getkernel.py. For Laws filter with run Getlawskernel.py with different laws filters mentioned in the report. In each of the file, save pca_params, feat, weights, biases and prediction vectors with appropriate names for combining later. Train prediction vectors needs to saved with name 'train' in it. Similarly for test prediction vectors. Code has already been provided. Just need to provide proper file names in the code.

Below is sequence of steps:
```
python Getkernel.py or python Getlawskernel.py
python Getfeature.py
python Getweight.py
python mnist_test.py
```

Additioanlly, make sure all files are in same folder including data.py (needed for loading data)

Once we have all train and test prediction vectors saved, then run ensemble.py to get final ensemble results

python ensemble.py
For comparision with Backpropogation based CNN, run the jupyter notebook BP-CNN.ipynb as it is.

## References

1. Code adapted from: https://github.com/davidsonic/Interpretable_CNNs_via_Feedforward_Design 
2. Kuo, C. C. J., Zhang, M., Li, S., Duan, J., & Chen, Y. (2019), Interpretable convolutional neural networks via feedforward design, Journal of Visual Communication and Image Representation.
3. Chen, Y., Yang, Y., Wang, W., & Kuo, C. C. J. (2019), Ensembles of feedforward-designed convolutional neural networks, arXiv preprint arXiv:1901.02154.
