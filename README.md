# Wildlife Image Classification using CNN

This repository demonstrates how to build a Convolutional Neural Network (CNN) to classify wildlife images (arctic fox, polar bear, and walrus) using Keras with TensorFlow backend. The project involves training a CNN from scratch, processing images, and visualizing the model's performance.

***Project Overview***

This project aims to classify wildlife images into three categories:

- Arctic Fox
- Polar Bear
- Walrus

It uses a CNN model with multiple layers of convolution and pooling to extract features from images, followed by fully connected layers for classification. The dataset is split into training and testing sets to evaluate the performance of the model.

***Files in the Repository***

- wildlife_cnn.py: The main script that loads the dataset, builds the CNN model, trains it, and evaluates performance.
- README.md: Provides an overview of the project and instructions on how to run it.

***Dataset***
The dataset used is structured as follows:

Data/
├── Wildlife/
    ├── train/
    │   ├── arctic_fox/    # Training images for arctic fox
    │   ├── polar_bear/    # Training images for polar bear
    │   └── walrus/        # Training images for walrus
    └── test/
        ├── arctic_fox/    # Test images for arctic fox
        ├── polar_bear/    # Test images for polar bear
        └── walrus/        # Test images for walrus
The images are loaded and resized to 224x224 pixels.

***Requirements***

To run this project, you will need:

- Python 3.7+
- TensorFlow / Keras
- NumPy
- Matplotlib
- OS Library for handling file paths

# To install the dependencies, run:
pip install tensorflow numpy matplotlib

***Model Architecture***

The CNN consists of the following layers:

- Conv2D Layer: 32 filters with 3x3 kernel, ReLU activation
- MaxPooling2D: 2x2 pooling
- Conv2D Layer: 128 filters with 3x3 kernel, ReLU activation
- MaxPooling2D: 2x2 pooling
- Conv2D Layer: 128 filters with 3x3 kernel, ReLU activation
- MaxPooling2D: 2x2 pooling
- Conv2D Layer: 128 filters with 3x3 kernel, ReLU activation
- MaxPooling2D: 2x2 pooling
- Flatten Layer
- Dense Layer: 1024 units, ReLU activation
- Dense Layer: 3 units (softmax activation for multi-class classification)
- The model is compiled with adam optimizer, categorical cross-entropy loss, and accuracy as the metric.

***Training and Validation***

The model is trained for 10 epochs with a batch size of 10. Both training and validation accuracy are tracked during the training process.

hist = model.fit(x_train, y_train_encoded, validation_data=(x_test, y_test_encoded), batch_size=10, epochs=10)

# Results

After training, you can visualize the accuracy over epochs by plotting both training and validation accuracy:

# Improvements

While the CNN performs reasonably well, achieving a decent accuracy, modern CNN models often achieve over 95% accuracy. To improve performance, consider:

Increasing Dataset Size: More training images would likely improve the model’s generalization.
Transfer Learning: Using pre-trained models such as VGG16 or ResNet50 can drastically boost accuracy without requiring large datasets or long training times.

# How to Run

***Clone the repository:***
- https://github.com/Tejasreebussa/Image-Classification.git
- cd Image-Classification

***Run the script:***
python Image Classification-CNN.py

Ensure the dataset is placed in the correct directory (Data/Wildlife/).