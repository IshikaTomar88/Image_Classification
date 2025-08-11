# Image Classification using Convolutional Neural Networks
This project aims to classify the images in the given dataset as cats or dogs using convolutional neural networks(CNN)

Approach and pipeline:
Refer to the report and code for the approach and implementation.

Results:
Results after training 18,000 images of cats and dogs:

number of epochs = 15
training data / validation data split = 80/20
MODEL
CONV 3x3 filter layers with batch norm - 32 x 64 x 96 x 96 x 64
Dense layers with drop out of 0.2 and 0.3 - 256 x 128 x 2
loss: 0.0638
accuracy: 0.9759
val_loss: 0.3255
val_accuracy: 0.9044
The model was tested on the images in the test1 folder. The performance of the model was very good and was able to predict the animals with 97-99% accuracy.

Plots for model accuracy and loss are following:

# Libraries
* Python 3.8+
* TensorFlow / PyTorch
* NumPy, Pandas
* Matplotlib, Seaborn
* CNN 
  
# Summary
 Load the CIFAR-10 dataset.
 Normalize training and test data.
 Change labels from integer to categorical.
 Build the model.
 Compile the model.
 Train the model.
 Save the model.
 Classify new test image using the trained model.


* Supports custom datasets in this format:


