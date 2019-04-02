# Motivation
This code is an example of how to use a Faster-RCNN to detect and read numbers on pictures.
You will find in **Datasets/img/** a set of about 2000 pictures and their labels in **Datasets/labels/**.

These labels follow the pascal VOC format and where created using **labelImg** (https://github.com/tzutalin/labelImg). 

The FasterRCNN is implemented using the excellent example from Tensorpack. A few modifications have been made (the most important one being not using data augmentation by flipping all examples, as it would create problems with the digits 2 and 5).

The code is not clean or perfect as it was implemented for a Hackathon in 2 days (including labelling, which took most of the available time). This code had the best accuracy vs. all other teams, with an average accuracy of **92%** on the final number reading (see **1. Reading.ipynb** for the evaluation).

# Installation
Requirements are not provided, however here is the list of libraries used:
~~~
pickle
pandas
tables
tensorpack (github)
cocoapi (github)
objec-detection-metrics (github)
numpy
tqdm
scikit-learn
cv2
jupyter
tensorflow-gpu
~~~
These should be enough to run the code. However, please note the this code will not run under Windows, and I highly recommend Linux with a decent GPU.

# How to use
Place all your pictures in Datasets/img/, all labels in PascalVOC format in Datasets/labels.
Run 0. VOC 2 OBF.ipynb
Train the faster RCNN (refer to https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN)
The output of the test set is saved in **out.pickle**.

On the given examples, training took about 30 minutes on one Tesla V100. (Training was stopped early).
# Examples
Here are 2 pictures randomly selected on google to prove the efficiency of this method:
![First example](https://github.com/faameunier/digit_recognition/img/google_random_test.png)
![Second example](https://github.com/faameunier/digit_recognition/img/google_random_test_2.png)




