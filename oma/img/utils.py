from os import listdir, remove, rename
from os.path import isfile, join
import cv2
from shutil import copyfile


def remove_invalid_formats(path):
    """equivalent to rm path/*.gif"""
    images = [path + f for f in listdir(path) if isfile(join(path, f))]
    for data in images:
        if data.lower().endswith(".gif"):
            remove(data)


def rename_blobs(path, asset):
    """a little helper to remove the asset name from pictures"""
    images = [f for f in listdir(path) if isfile(join(path, f))]
    n_asset = len(asset)
    for data in images:
        rename(path + data, path + data[n_asset:])


def save_images(folder_out, names, imgs):
    """saves a batch of images"""
    for i in range(len(names)):
        cv2.imwrite(folder_out + names[i], imgs[i])


def load_images(folder):
    """loads a batch of images"""
    paths = [folder + f for f in listdir(folder) if isfile(join(folder, f))]
    paths.sort()
    images = []
    for p in paths:
        images.append(cv2.imread(p))
    return (paths, images)


def copy_images(paths, folder_out):
    """copy the paths to a new folder"""
    for p in paths:
        copyfile(p, folder_out + p)
