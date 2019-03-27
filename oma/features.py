import mahotas
import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


def fd_haralick(image):    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick


def fd_histogram(image, mask=None):
    features_hist = []
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    bins=32
    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [bins], [0, 256])
        features_hist.extend(hist)
        cv2.normalize(hist, hist)
    return(np.array(features_hist).flatten())

def feature_extractor(image):
    global_feature = np.hstack([fd_histogram(image), fd_haralick(image), fd_hu_moments(image)])
    return (global_feature)
