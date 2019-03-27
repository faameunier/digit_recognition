import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from multiprocessing import Pool, cpu_count
from functools import partial
import tensorflow as tf


def resize(img, px):
    """resize an image to a square of px size"""
    return cv2.resize(img, (px, px))


def similarity_corr(imgs):
    """compute similarity matrix using correlation"""
    temp = np.array(imgs).astype(np.float16)
    temp = temp.reshape(len(temp), -1)
    temp = np.corrcoef(temp)
    temp = np.nan_to_num(temp)
    np.fill_diagonal(temp, 0)
    return np.triu(temp)


def similarity_cos(emb):
    """compute similarity matrix using cosine sim with GPU computing if available"""
    tensor_in = tf.placeholder(tf.float16, shape=emb.shape)
    norms = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(tensor_in), 1))
    norms_x = tf.expand_dims(norms, -1)
    norms_inv = tf.math.reciprocal(norms_x)
    tensor = tf.math.multiply(tensor_in, norms_inv)
    output = tf.matmul(tensor, tf.transpose(tensor))
    with tf.Session() as sess:
        temp = sess.run(output, feed_dict={tensor_in: emb})
    np.fill_diagonal(temp, 0)
    return np.triu(temp)


def similarity_cos_np(emb):
    """compute similarity matrix using cosine sim via numpy"""
    temp = emb.copy()
    norms = np.sqrt(np.sum(temp**2, axis=1))
    temp = np.multiply(temp, 1 / norms[:, np.newaxis])
    temp = np.matmul(temp, temp.T)
    np.fill_diagonal(temp, 0)
    return np.triu(temp)


def _preprocess_img(path, BW=False, k=8, px=128):
    """Preprocess an image

    Loads the image into memory, resizes it to
    px * px size, convert it to black and white if necessary
    and performs color simplification to k colors.

    Arguments:
        path {str} -- Path to the image

    Keyword Arguments:
        BW {bool} -- Black and white output (default: {False})
        k {number} -- number of colors (default: {8})
        px {number} -- final size (default: {128})

    Returns:
        np.array -- preprocessed image in BGR format
    """
    im = cv2.imread(path)
    if BW:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if k != 0:
        im = color_simp(im, BW, k)
    return resize(im, px)


def preprocess(folder_in, BW=False, k=8, px=128):
    """Preprocess a whole folder

    See _preprocess_img for details. using
    multiplrocessing.

    Arguments:
        folder_in {str} -- folder to preprocess

    Keyword Arguments:
        BW {bool} -- Black and white output (default: {False})
        k {number} -- number of colors (default: {8})
        px {number} -- final size (default: {128})

    Returns:
        list, list -- list of images and list of paths in the same order
    """
    images = [folder_in + f for f in listdir(folder_in) if isfile(join(folder_in, f))]
    images.sort()
    res = []
    n = len(images)
    fun = partial(_preprocess_img, BW=BW, k=k, px=px)
    with Pool(cpu_count()) as p:
        with tqdm(total=n) as pbar:
            for r in p.imap(fun, images):
                pbar.update()
                res.append(r)
    return images, res


def color_simp(image, BW=False, k=8):
    """Perform a color simplification

    Given an image, reduces the number of colors
    to k using clustering.

    Arguments:
        image {np.array} -- original image

    Keyword Arguments:
        BW {bool} -- Black and White (default: {False})
        k {number} -- number of colors (default: {8})

    Returns:
        np.array -- simplified image
    """
    (h, w) = image.shape[:2]
    k_max = len(np.unique(image.flatten()))
    k = min(k_max, k)
    if BW:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = MiniBatchKMeans(n_clusters=k)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    quant = quant.reshape((h, w, 3))
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    if BW:
        quant = cv2.cvtColor(quant, cv2.COLOR_BGR2GRAY)
    return quant


def unique(paths, similarities, threshold=0.97):
    """Guess unique images

    Returns a list of 'unique' images based
    on a similarity matrix. Two images are considered
    different if and only if their similarity is below the
    threshold.

    Be careful : paths and similarities must be in the same order.

    Arguments:
        paths {list} -- list of paths
        similarities {np.array} -- Similarity matrix

    Keyword Arguments:
        threshold {number} -- threshold used to guess uniqueness (default: {0.97})

    Returns:
        np.array -- list of unique paths
    """
    res = []
    for row in similarities:
        res += np.argwhere(np.abs(row) >= threshold).flatten().tolist()
    res = set(list(range(len(paths)))) - set(res)
    res = list(res)
    return np.array(paths)[res]
