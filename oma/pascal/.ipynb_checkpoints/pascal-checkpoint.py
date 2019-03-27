import xml.etree.ElementTree as ET
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split


def read(path):
    """Read XML

    Read a xml file

    Arguments:
        path {str} -- path to xml

    Returns:
        xml.etree.ElementTree -- XML ET API
    """
    tree = ET.parse(path)
    return tree


def get_BB(obj):
    """Get bounding box

    Get bounding box of an object from XML tree

    Arguments:
        obj {ElementTree} -- object tree

    Returns:
        dict -- dictionnary with the bouding box
    """
    BB = {
        'xmin': None,
        'xmax': None,
        'ymin': None,
        'ymax': None
    }
    temp = obj.find("bndbox")
    BB['xmin'] = int(temp.find('xmin').text)
    BB['xmax'] = int(temp.find('xmax').text)
    BB['ymin'] = int(temp.find('ymin').text)
    BB['ymax'] = int(temp.find('ymax').text)
    return BB


def get_category(obj):
    """Get category

    Get ctageory of an object

    Arguments:
        obj {ElementTree} -- object tree

    Returns:
        str -- object's category
    """
    return obj.find("name").text


def get_path(tree):
    """Get path

    Get path of image

    Arguments:
        tree {ElementTree} -- pascal voc tree

    Returns:
        str -- path to the image
    """
    return tree.find("path").text


def set_path(tree, folder):
    """Set path of image

    Usefull to correct absolute paths
    to relative paths.

    Arguments:
        tree {ElementTree} -- pascal voc tree
        folder {path} -- relative folder
    """
    filename = tree.find("filename").text
    if not filename.endswith(".jpg"):
        filename += ".jpg"
    tree.find("path").text = folder + filename


def save(tree, path):
    """Save tree"""
    tree.write(path)


def correct_all_paths(folder_labels, folder_img):
    """Correct absolute paths

    Read all XML files in folder_labels and set images
    path to relative paths toward folder dolder_img.

    Arguments:
        folder_labels {str} -- folder containing labels
        folder_img {str} -- folder containing images
    """
    labels = [folder_labels + f for f in listdir(folder_labels) if isfile(join(folder_labels, f))]
    for label in tqdm(labels):
        try:
            tree = read(label)
            set_path(tree, folder_img)
            tree.write(label)
        except Exception as e:
            print(label)
            raise e


def get_all_objects(folder_labels):
    """Get all objects

    Read all xml files in the folder and
    returns all object present in all
    images in a DataFrame with category
    and bounding box.

    Arguments:
        folder_labels {str} -- path to labels

    Returns:
        pd.DataFrame -- df with object details
    """
    cols = ["image_path", "name", "category", "object_n", "xmin", "xmax", "ymin", "ymax"]
    df = pd.DataFrame(columns=cols)
    labels = [folder_labels + f for f in listdir(folder_labels) if isfile(join(folder_labels, f))]
    for label in tqdm(labels):
        tree = read(label)
        path = get_path(tree)
        name = tree.find("filename").text
        k = 0
        for obj in tree.findall("object"):
            cat = get_category(obj)
            BB = get_BB(obj)
            n = k
            df.loc[len(df)] = [path, name, cat, n, BB["xmin"], BB["xmax"], BB["ymin"], BB["ymax"]]
            k += 1
    return df


def objects_to_OBF(objects_df, output_folder, item_db_path):
    """Convert an Object DF to OBF format

    Products are cropped and saved in the output folder
    while a new df is built in the item_db fashion. Not all
    columns are present but enough to provide basic compatibility.
    This new item_db is saved to the specified path.

    Arguments:
        objects_df {pd.DataFrame} -- objects df (output from get_all_objects)
        output_folder {str} -- folder where images are saved
        item_db_path {str} -- path for the item_db dataframe (hdf format, obf compatible)
    """
    cols = ["sku", "category"]
    df = pd.DataFrame(columns=cols)

    def treat_row(row):
        img = cv2.imread(row["image_path"])
        product = img[row["ymin"]:row["ymax"], row["xmin"]:row["xmax"]]
        sku = str(row["name"][:-4]) + str(row["object_n"])
        df.loc[len(df)] = [sku, row["category"]]
        cv2.imwrite(output_folder + sku + ".jpg", product)

    tqdm.pandas()
    objects_df.progress_apply(treat_row, axis=1)
    df.to_hdf(item_db_path, key="df")


def objects_to_FRCNN(objects_df, df_path, test_size=0.2, val_size=0.2):
    temp = objects_df[["image_path", "xmin", "ymin", "xmax", "ymax", "category"]]
    ids = list(temp.index)
    train, test = train_test_split(ids, test_size=test_size)
    train, val = train_test_split(train, test_size=val_size)
    types = pd.DataFrame({'type': [None]})
    temp = temp.join(types)
    temp.loc[train, "type"] = "train"
    temp.loc[test, "type"] = "test"
    temp.loc[val, "type"] = "val"
    temp.to_hdf(df_path, key="df")
    return temp

def clean_cat(objects_df, categories):
    temp = objects_df.merge(categories, how='left', left_on='category', right_on='OLD')
    temp['category'] = temp['NEW']
    temp = temp.drop(columns=['OLD', 'NEW'])
    temp = temp[temp['category'].notnull()]
    return temp.drop_duplicates()
