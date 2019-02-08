# Optimising a Recurrent Neural Network with hyperopt and tensorboardX

<img align="right" src="tensorboardXample.png" width=350 height=400/>

In this project I worked on tuning a recurrent neural networks hyper parameters using the popular optimisation package [hyperopt](https://github.com/hyperopt/hyperopt).

The data used was from the [Kaggle Quora Competition](https://www.kaggle.com/c/quora-insincere-questions-classification), a competition where the aim was to build a insincere question classifier. The inspiration to optimise a neural network for this task came from the competition constraints, the whole network must be able to train and predict within 2 hours on a kaggle docker image with GPU enabled. This meant that a highly optimised neural net architecture would be a sure way to perform well in this competition. Hyperopt is not the only optimisation framework I could have used, and I am eager to try out optuna and others in the near future.

TensorboardX is one of the most useful tools for visualising scalars / data during training, and when using an optimiser this becomes crucial to monitor the best performing candidate parameters for a network. For the sake of simplicity I have only optimised over a few parameters here, however it is easy to see the potential in using an optimizer such as hyperopt to tune your network.

To test the optimiser, first download the data off Kaggle into the input folder in this repository. Then run the python script preprocess.py, and main.py. 

NOTE: the download requires 6GB of space mostly due to the large embedding files, however to run this optimiser you will only need the glove embeddings and can avoid downloading the others. 

