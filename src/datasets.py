# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""

import os
import sys
import csv
import tarfile
import shutil
import hashlib
from tqdm import tqdm
from urllib.request import urlretrieve
from urllib.error import URLError
from urllib.error import HTTPError
csv.field_size_limit(sys.maxsize)
DATA_FOLDER = "datasets"


def _progress(count, block_size, total_size):
    rate = float(count * block_size) / float(total_size) * 100.0
    sys.stdout.write("\r>> Downloading {:.1f}%".format(rate))


def get_file(fname, origin, untar=False, md5_hash=None, cache_subdir='datasets', check_certificate=True):
    """Downloads a file from a URL if it not already in the cache.

    Passing the MD5 hash will verify the file after download
    as well as if it is already present in the cache.

    # Arguments
        fname: name of the file
        origin: original URL of the file
        untar: boolean, whether the file should be decompressed
        md5_hash: MD5 hash of the file for verification
        cache_subdir: directory being used as the cache

    # Returns
        Path to the downloaded file
    """
    datadir = cache_subdir
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + '.tar.gz'
    else:
        fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath):
        # File found; verify integrity if a hash was provided.
        if md5_hash is not None:
            if not validate_file(fpath, md5_hash):
                print('A local file was found, but it seems to be '
                      'incomplete or outdated.')
                download = True
    else:
        download = True

    if download:
        print('Downloading data from', origin)
        error_msg = 'URL fetch failure on {}: {} -- {}'

        if not check_certificate:
            import requests
            r = requests.get(origin, verify=False)
            with open(fpath, 'wb') as fd:
                fd.write(r.content)
        else:
            try:
                try:
                    urlretrieve(origin, fpath, _progress)
                    sys.stdout.flush()
                except URLError as e:
                    raise Exception(error_msg.format(origin, e.errno, e.reason))
                except HTTPError as e:
                    raise Exception(error_msg.format(origin, e.code, e.msg))
            except (Exception, KeyboardInterrupt) as e:
                if os.path.exists(fpath):
                    os.remove(fpath)
                raise

    if untar:
        if not os.path.exists(untar_fpath):
            print('Untaring file...')
            tfile = tarfile.open(fpath, 'r:gz')
            try:
                tfile.extractall(path=datadir)
            except (Exception, KeyboardInterrupt) as e:
                if os.path.exists(untar_fpath):
                    if os.path.isfile(untar_fpath):
                        os.remove(untar_fpath)
                    else:
                        shutil.rmtree(untar_fpath)
                raise
            tfile.close()
        return untar_fpath

    return fpath


def validate_file(fpath, md5_hash):
    """Validates a file against a MD5 hash.

    # Arguments
        fpath: path to the file being validated
        md5_hash: the MD5 hash being validated against

    # Returns
        Whether the file is valid
    """
    hasher = hashlib.md5()
    with open(fpath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    if str(hasher.hexdigest()) == str(md5_hash):
        return True
    else:
        return False

class Demo(object):
    def __init__(self):
        self.data_name = "demo"  # ag_news
        self.data_folder = "{}/{}/raw".format(DATA_FOLDER, self.data_name)
        self.n_classes = 2
        self.epoch_size = 5000

        for f in ["classes.txt", "demo.csv", "char2byte.pkl"]:
            if not os.path.exists(os.path.join(self.data_folder, f)):
                print("Necessary file for demo does not exist.")

    def _generator(self, filename):
        with open(filename, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f, fieldnames=['label', 'title', 'description'], quotechar='"')
            for line in reader:
                sentence = "{}".format(line['description'])
                label = int(line['label']) - 1
                # print(sentence, label)
                yield sentence, label

    def load_test_data(self):
        return self._generator(os.path.join(self.data_folder, "demo.csv"))

class Noah(object):
    def __init__(self):
        self.url = "https://homes.cs.washington.edu/~nasmith/temp/challenge-data.tgz"
        self.data_name = "noah" # ag_news
        self.data_folder = "{}/{}/raw".format(DATA_FOLDER, self.data_name)
        self.n_classes = 2
        self.epoch_size = 5000

        # Check if relevant files are in the folder_path
        if os.path.exists(self.data_folder):
            for f in ["classes.txt", "test.csv", "train.csv"]:
                if not os.path.exists(os.path.join(self.data_folder, f)):
                    self._ = get_file(self.data_name, origin=self.url, untar=True, cache_subdir=self.data_folder)
        else:
            self._ = get_file(self.data_name, origin=self.url, untar=True, cache_subdir=self.data_folder)

    def _generator(self, filename, num):
        with open(filename, mode='r', encoding='utf-8') as f:
            counter = 0
            reader = csv.DictReader(f, fieldnames=['label', 'title', 'description'], quotechar='"')
            for line in reader:
                if counter >= num:
                    break
                sentence = "{}".format(line['description'])
                label = int(line['label']) - 1
                counter += 1
                # print(sentence, label)
                yield sentence, label

    def load_train_data(self):
        return self._generator(os.path.join(self.data_folder, "train.csv"), 100000)

    def load_test_data(self):
        return self._generator(os.path.join(self.data_folder, "test.csv"), 200000)

class AgNews(object):
    """
    credit goes to Xiang Zhang:
    https://scholar.google.com/citations?hl=en&user=n4QjVfoAAAAJ&view_op=list_works&sortby=pubdate
    """
    def __init__(self):

        self.url = "https://s3.eu-west-2.amazonaws.com/ardalan.mehrani.datasets/ag_news.tar.gz"
        self.data_name = os.path.basename(self.url).split(".")[0] # ag_news
        self.data_folder = "{}/{}/raw".format(DATA_FOLDER, self.data_name)
        self.n_classes = 4        
        self.epoch_size = 5000

        # Check if relevant files are in the folder_path
        if os.path.exists(self.data_folder):
            for f in ["classes.txt", "readme.txt", "test.csv", "train.csv"]:
                if not os.path.exists(os.path.join(self.data_folder, f)):
                    self._ = get_file(self.data_name, origin=self.url, untar=True, cache_subdir=self.data_folder)
        else:
            self._ = get_file(self.data_name, origin=self.url, untar=True, cache_subdir=self.data_folder)

    def _generator(self, filename):

        with open(filename, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f, fieldnames=['label', 'title', 'description'], quotechar='"')
            for line in reader:
                sentence = "{} {}".format(line['title'], line['description'])
                label = int(line['label']) - 1
                yield sentence, label

    def load_train_data(self):
        return self._generator(os.path.join(self.data_folder, "train.csv"))

    def load_test_data(self):
        return self._generator(os.path.join(self.data_folder, "test.csv"))

class YelpReview(object):
    """
    credit goes to Xiang Zhang:
    https://scholar.google.com/citations?hl=en&user=n4QjVfoAAAAJ&view_op=list_works&sortby=pubdate
    """
    def __init__(self):

        self.url = "https://s3.eu-west-2.amazonaws.com/ardalan.mehrani.datasets/yelp_review_full.tar.gz"
        self.data_name = os.path.basename(self.url).split(".")[0] # ag_news
        self.data_folder = "{}/{}/raw".format(DATA_FOLDER, self.data_name)
        self.n_classes = 5
        self.epoch_size = 5000

        # Check if relevant files are in the folder_path
        if os.path.exists(self.data_folder):
            for f in ["readme.txt", "test.csv", "train.csv"]:
                if not os.path.exists(os.path.join(self.data_folder, f)):
                    self._ = get_file(self.data_name, origin=self.url, untar=True, cache_subdir=self.data_folder)
        else:
            self._ = get_file(self.data_name, origin=self.url, untar=True, cache_subdir=self.data_folder)

    @staticmethod
    def _generator(filename):

        with open(filename, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f, fieldnames=['label', 'title', 'description'], quotechar='"')
            for line in reader:
                sentence = "{} {}".format(line['title'], line['description'])
                label = int(line['label']) - 1
                yield sentence, label

    def load_train_data(self):
        return self._generator(os.path.join(self.data_folder, "train.csv"))

    def load_test_data(self):
        return self._generator(os.path.join(self.data_folder, "test.csv"))


class YelpPolarity(object):
    """
    credit goes to Xiang Zhang:
    https://scholar.google.com/citations?hl=en&user=n4QjVfoAAAAJ&view_op=list_works&sortby=pubdate
    """
    def __init__(self):

        self.url = "https://s3.eu-west-2.amazonaws.com/ardalan.mehrani.datasets/yelp_review_polarity.tar.gz"
        self.data_name = os.path.basename(self.url).split(".")[0] # ag_news
        self.data_folder = "{}/{}/raw".format(DATA_FOLDER, self.data_name)
        self.n_classes = 2        
        self.epoch_size = 5000

        # Check if relevant files are in the folder_path
        if os.path.exists(self.data_folder):
            for f in ["readme.txt", "test.csv", "train.csv"]:
                if not os.path.exists(os.path.join(self.data_folder, f)):
                    self._ = get_file(self.data_name, origin=self.url, untar=True, cache_subdir=self.data_folder)
        else:
            self._ = get_file(self.data_name, origin=self.url, untar=True, cache_subdir=self.data_folder)

    @staticmethod
    def _generator(filename):

        with open(filename, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f, fieldnames=['label', 'title', 'description'], quotechar='"')
            for line in reader:
                sentence = "{} {}".format(line['title'], line['description'])
                label = int(line['label']) - 1
                yield sentence, label

    def load_train_data(self):
        return self._generator(os.path.join(self.data_folder, "train.csv"))

    def load_test_data(self):
        return self._generator(os.path.join(self.data_folder, "test.csv"))


def load_datasets(names=["noah"]):
    """
    Select datasets based on their names

    :param names: list of string of dataset names
    :return: list of dataset object
    """

    datasets = []
    if 'demo' in names:
        datasets.append(Demo())
    if 'noah' in names:
        datasets.append(Noah())
    if 'ag_news' in names:
        datasets.append(AgNews())    
    if 'yelp_review' in names:
        datasets.append(YelpReview())
    if 'yelp_polarity' in names:
        datasets.append(YelpPolarity())
    
    return datasets


if __name__ == "__main__":

    # names = [
    #     'ag_news',
    #     'yelp_review',
    #     'yelp_polarity',
    # ]

    names = ['noah']

    for name in names:
        print("name: {}".format(name))
        dataset = load_datasets(names=[name])[0]
        
        # train data generator
        gen = dataset.load_train_data()
        sentences, labels = [], []
        for sentence, label in tqdm(gen):
            sentences.append(sentence)
            labels.append(label)
        print(" train: (sentences,labels) = ({}/{})".format(len(sentences), len(labels)))

        # test data generator
        gen = dataset.load_test_data()
        sentences, labels = [], []
        for sentence, label in tqdm(gen):
            sentences.append(sentence)
            labels.append(label)
        print(" train: (sentences,labels) = ({}/{})".format(len(sentences), len(labels)))