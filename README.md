# Image Classification using Convolutional Neural Networks
This project aims to classify the images in the given dataset as cats or dogs using convolutional neural networks(CNN)

<img width="1255" height="300" alt="Screenshot (35)" src="https://github.com/user-attachments/assets/6455939e-9232-408d-9c7b-d779d003ba92" />

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

<img width="699" height="774" alt="Screenshot (36)" src="https://github.com/user-attachments/assets/02888fe9-e042-487a-af55-2b9f95f9e6fc" />

<img width="693" height="780" alt="Screenshot (37)" src="https://github.com/user-attachments/assets/e415cf09-dede-4b94-9d28-75515987323d" />

 <img width="699" height="785" alt="Screenshot (38)" src="https://github.com/user-attachments/assets/cea7787b-5e7d-43fa-b684-0dc953716ab4" />

# Libraries
* Python 3.8+
* TensorFlow / PyTorch
* NumPy, Pandas
* Matplotlib, Seaborn
* CNN

# Patform :
Google colab

# Summary
* Load the CIFAR-10 dataset.
* Normalize training and test data.
* Change labels from integer to categorical.
* Build the model.
* Compile the model.
* Train the model.
* Save the model.
* Classify new test image using the trained model.

# Output/ Result:
<img width="1190" height="413" alt="Screenshot (39)" src="https://github.com/user-attachments/assets/68c26e84-4321-4f81-8b5e-bbea671ec125" />


