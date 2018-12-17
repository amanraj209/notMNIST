# notMNIST
Performed binary classification using CNN on notMNIST dataset

# About the dataset
There are 10 classes, with letters A-J taken from different fonts. Here are some examples of letter "A". Judging by the examples, one would expect this to be a harder task than MNIST. This seems to be the case -- logistic regression on top of stacked auto-encoder with fine-tuning gets about 89% accuracy whereas same approach gives got 98% on MNIST. Dataset consists of small hand-cleaned part, about 19k instances, and large uncleaned dataset, 500k instances.

# The model
This repository uses a CNN with 4 convolution layers having two maxpooling layers. Then, it uses two dense layers with batch normalization and dropout. We got an accuracy of 94% on the test set and 92% on the validation set.

The repository takes into account the following things:

1. It constantly makes checkpoints of the epochs and can resume after stopping. For example, if you run the code for 20 eppchs and then run it again for 40 epochs, then it will start from epoch number 21 by using the checkpoints stored.

2. It also uses **tfrecords** for fast data storing and retrieval of large number of images. 

# To run
1. Clone the repository

2. Edit the file /configs/config.json -> change the path of **data_numpy_pkl** to your own path. Change the batch_size, number of epochs and learning rate according to your model. You can change these hyper parameters from here.

3. **cd** (Change directory) into **mains/**. Run the command: **python main.py -c "absolute path of config.json"**

4. You will find the result in experiments folder. The tfsummary will be present int the summary folder and you can view it using tensorboard.
