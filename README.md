# Fuzzy-Pooling
This project is an implementation of the novel fuzzy pooling technique proposed in the paper Fuzzy Pooling (https://arxiv.org/pdf/2202.08372) by Dimitrios E. Diamantis and Dimitris K. Iakovidis.

## Implementation and Evaluation
The implementation is done using TensorFlow and Keras. First we created a custom FuzzyPoolingLayer in Keras, designed to replace traditional pooling layers in a CNN. This layer
performs the Fuzzy Pooling operation in following 3 steps: fuzzification, aggregation and defuzzification, on the feature maps produced by a convolutional layer.

The CNN architecture was structured as follows: a Conv2D layer with ReLU activation, followed by the FuzzyPoolingLayer, and finally a Dense output layer for classification.  The model was compiled using an Adam optimizer with standard training parameters (batch size of 32, learning rate of 0.001) for effective training on the selected datasets.

To assess the effectiveness of Fuzzy Pooling, its performance was comapred with traditional pooling methods, Max Pooling and Average Pooling. Each model variant was trained on the MNIST, Fashion-MNIST, and CIFAR-10 datasets and their classification accuracy was compared.

## Results
Results indicated that Fuzzy Pooling outperformed traditional pooling methods in accuracy, especially on complex datasets like CIFAR-10, which contains more diverse and challenging images. This improvement demonstrates Fuzzy Pooling’s ability to retain meaningful features in the presence of noise and variability.
