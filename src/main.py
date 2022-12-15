import os
import re
import torch
import lmdb
import pickle
import itertools
import argparse
import numpy as np
import torch.nn as nn
import pickle as pkl
import collections
import time

from tqdm import tqdm
from collections import Counter
from sklearn import utils, metrics
from src.datasets import load_datasets

from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from src.net import SVDCNN

# multiprocessing workaround
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (1024, rlimit[1]))

# Random seed
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

def count_parameters(model):
    total_param = 0
    total_param_conv = 0
    total_param_fc = 0

    for name, param in model.named_parameters():
            num_param = np.prod(param.size())
            # if param.dim() > 1:
            #     print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)                
            # else:
            #     print(name, ':', num_param)

            if ('fc' in name ):
                total_param_fc += num_param
            else:
                total_param_conv += num_param
           
            total_param += num_param

    return total_param, total_param_conv, total_param_fc


def get_args():
    parser = argparse.ArgumentParser("""Squeezed Very Deep Convolutional Neural Networks for Text Classification""")
    parser.add_argument("--dataset", type=str, default='ag_news')
    parser.add_argument("--model_folder", type=str, default="models/svdcnn/ag_news")
    parser.add_argument("--data_folder", type=str, default="datasets/ag_news/svdcnn")
    parser.add_argument("--depth", type=int, choices=[9, 17, 29], default=9, help="Depth of the network tested in the paper (9, 17, 29)")
    parser.add_argument("--maxlen", type=int, default=1024)
    parser.add_argument("--shortcut", type=bool)
    parser.add_argument("--batch_size", type=int, default=128, help="number of example read by the gpu")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_halve_interval", type=float, default=100, help="Number of iterations before halving learning rate")
    parser.add_argument("--snapshot_interval", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--nthreads", type=int, default=4)
    args = parser.parse_args()
    return args

class Preprocessing():

    def __init__(self, lowercase=True):
        self.lowercase = lowercase

    def transform(self, sentences):
        """
        sentences: list(str) 
        output: list(str)
        """
        return [s.lower() for s in sentences]

class ByteVectorizer():
    def __init__(self, char_dict=None, max_features=175, maxlen=1024, padding='pre', truncating='pre', dataset="noah"):
        self.char_dict = char_dict
        self.max_features = max_features
        self.maxlen = maxlen
        self.padding = padding
        self.truncating = truncating
        self.char_counter = Counter()
        self.dataset = dataset

        self.n_transform = 0

        if self.char_dict:
            self.n_transform += 1

    def partial_fit(self, sentences):
        """
        sentences: list of list
        """
        for sentence in sentences:
            self.char_counter.update(sentence)

    def transform(self, sentences):
        """
        sentences: list of string
        list of review, review is a list of sequences, sequences is a list of int
        """
        DATA_FOLDER = "datasets"
        # keys = list("”abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’\"/| #$%ˆ&*˜‘+=<>()[]{}")
        data_folder = "{}/{}/raw".format(DATA_FOLDER, self.dataset)
        keys = []
        with open(f"{data_folder}/bytes.txt", "r") as f:
            for line in f:
                line = line.replace("\n", "")
                if line != "":
                    keys.append(line)
        keys.append(" ")
        values = list(range(2, len(keys) + 2))
        self.char_dict = dict(zip(keys, values))
        self.char_dict["_pad_"] = 0
        self.char_dict["_unk_"] = 1

        assert self.char_dict, "No dictionnary to vectorize text \n-> call method build_dict \n-> or set a word_dict attribute \n first"

        sequences = []

        for sentence in sentences:
            seq = [self.char_dict.get(char, self.char_dict["_unk_"]) for char in sentence]

            if self.maxlen:
                length = len(seq)
                if self.truncating == 'pre':
                    seq = seq[-self.maxlen:]
                elif self.truncating == 'post':
                    seq = seq[:self.maxlen]

                if length < self.maxlen:

                    diff = np.abs(length - self.maxlen)

                    if self.padding == 'pre':
                        seq = [self.char_dict['_pad_']] * diff + seq

                    elif self.padding == 'post':
                        seq = seq + [self.char_dict['_pad_']] * diff
            sequences.append(seq)

        return sequences

    def get_params(self):
        params = vars(self)
        if 'char_counter' in params:
            del params['char_counter']
        return params


class CharVectorizer():
    def __init__(self,char_dict=None, max_features=69, maxlen=1024, padding='pre', truncating='pre'):
        self.char_dict = char_dict
        self.max_features = max_features
        self.maxlen = maxlen
        self.padding = padding
        self.truncating = truncating
        self.char_counter = Counter()

        self.n_transform = 0

        if self.char_dict:
            self.n_transform += 1
    
    def partial_fit(self, sentences):
        """
        sentences: list of list
        """
        for sentence in sentences:
            self.char_counter.update(sentence)

    def transform(self,sentences):
        """
        sentences: list of string
        list of review, review is a list of sequences, sequences is a list of int
        """

        values = list(range(2,self.max_features))
        keys = list("”abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’\"/| #$%ˆ&*˜‘+=<>()[]{}")

        self.char_dict = dict(zip(keys, values))
        self.char_dict["_pad_"] = 0
        self.char_dict["_unk_"] = 1

        assert self.char_dict, "No dictionnary to vectorize text \n-> call method build_dict \n-> or set a word_dict attribute \n first"
        

        sequences = []

        for sentence in sentences:
            seq = [self.char_dict.get(char, self.char_dict["_unk_"]) for char in sentence]
            
            if self.maxlen:
                length = len(seq)
                if self.truncating == 'pre':
                    seq = seq[-self.maxlen:]
                elif self.truncating == 'post':
                    seq = seq[:self.maxlen]

                if length < self.maxlen:

                    diff = np.abs(length - self.maxlen)

                    if self.padding == 'pre':
                        seq = [self.char_dict['_pad_']] * diff + seq

                    elif self.padding == 'post':
                        seq = seq + [self.char_dict['_pad_']] * diff
            sequences.append(seq)                

        return sequences        
    
    def get_params(self):
        params = vars(self)
        if 'char_counter' in params:
            del params['char_counter'] 
        return params


class TupleLoader(Dataset):

    def __init__(self, path=""):
        self.path = path

        self.env = lmdb.open(path, max_readers=4, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)

    def __len__(self):
        return list_from_bytes(self.txn.get('nsamples'.encode()))[0]

    def __getitem__(self, i):
        xtxt = list_from_bytes(self.txn.get(('txt-%09d' % i).encode()), int)
        lab = list_from_bytes(self.txn.get(('lab-%09d' % i).encode()), int)[0]
        return xtxt, lab


def get_metrics(cm, list_metrics):
    """Compute metrics from a confusion matrix (cm)
    cm: sklearn confusion matrix
    returns:
    dict: {metric_name: score}

    """
    dic_metrics = {}
    total = np.sum(cm)

    if 'accuracy' in list_metrics:
        out = np.sum(np.diag(cm))
        dic_metrics['accuracy'] = out/total

    if 'pres_0' in list_metrics:
        num = cm[0, 0]
        den = cm[:, 0].sum()
        dic_metrics['pres_0'] =  num/den if den > 0 else 0

    if 'pres_1' in list_metrics:
        num = cm[1, 1]
        den = cm[:, 1].sum()
        dic_metrics['pres_1'] = num/den if den > 0 else 0

    if 'recall_0' in list_metrics:
        num = cm[0, 0]
        den = cm[0, :].sum()
        dic_metrics['recall_0'] = num/den if den > 0 else 0

    if 'recall_1' in list_metrics:
        num = cm[1, 1]
        den = cm[1, :].sum()
        dic_metrics['recall_1'] =  num/den if den > 0 else 0

    return dic_metrics


def train(epoch,net,dataset,device,msg="val/test",optimize=False,optimizer=None,scheduler=None,criterion=None):
    
    net.train() if optimize else net.eval()

    epoch_loss = 0
    nclasses = len(list(net.parameters())[-1])
    cm = np.zeros((nclasses,nclasses), dtype=int)

    with tqdm(total=len(dataset),desc="Epoch {} - {}".format(epoch, msg)) as pbar:
        for iteration, (tx, ty) in enumerate(dataset):

            data = (tx, ty)
            # print("data: ", data)
            data = [x.to(device) for x in data]

            if optimize:
                optimizer.zero_grad()

            out = net(data[0])
            ty_prob = F.softmax(out, 1) # probabilites
            # print("ty_prob: ", ty_prob)
            #metrics
            y_true = data[1].detach().cpu().numpy()
            # print("y_true: ", y_true)
            y_pred = ty_prob.cpu().max(1)
            # print("y_pred: ", y_pred)
            y_pred = y_pred[1]

            cm += metrics.confusion_matrix(y_true, y_pred, labels=range(nclasses))
            dic_metrics = get_metrics(cm, list_metrics)

            loss =  criterion(out, data[1])
            epoch_loss += loss.item()
            dic_metrics['logloss'] = epoch_loss/(iteration+1)

            if optimize:
                loss.backward()
                optimizer.step()
                dic_metrics['lr'] = optimizer.state_dict()['param_groups'][0]['lr']

            pbar.update(1)
            pbar.set_postfix(dic_metrics)

    if scheduler:
        scheduler.step()


def save(net, txt_dict, path):
    """
    Saves a model's state and it's embedding dic by piggybacking torch's save function
    """
    dict_m = net.state_dict()
    dict_m["txt_dict"] = txt_dict
    torch.save(dict_m,path)


def list_to_bytes(l):
    return np.array(l).tobytes()


def list_from_bytes(string, dtype=np.int):
    return np.frombuffer(string, dtype=dtype)


if __name__ == "__main__":

    opt = get_args()
    print("\nparameters: {}\n".format(vars(opt)))
    
    os.makedirs(opt.model_folder, exist_ok=True)
    os.makedirs(opt.data_folder, exist_ok=True)

    dataset = load_datasets(names=[opt.dataset])[0]
    dataset_name = dataset.__class__.__name__
    n_classes = dataset.n_classes
    print("dataset: {}, n_classes: {}".format(dataset_name, n_classes))

    
    variables = {
        'train': {'var': None, 'path': "{}/train.lmdb".format(opt.data_folder)},
        'test': {'var': None, 'path': "{}/test.lmdb".format(opt.data_folder)},
        'txt_dict': {'var': None, 'path': "{}/txt_dict.pkl".format(opt.data_folder)},
    }

    # check if datasets exis
    all_exist = True if os.path.exists(variables['txt_dict']['path']) else False
    if all_exist:
        # print("  - Loading: {}".format(variables['txt_dict']['path']))
        variables['txt_dict']['var'] = pkl.load(open(variables['txt_dict']['path'],"rb"))
        n_tokens = len(variables['txt_dict']['var']['char_dict'])
    else:
        print("Creating datasets")
        tr_sentences = [txt for txt,lab in tqdm(dataset.load_train_data(), desc="counting train samples")]
        te_sentences = [txt for txt,lab in tqdm(dataset.load_test_data(), desc="counting test samples")]
            
        n_tr_samples = len(tr_sentences)
        n_te_samples = len(te_sentences)

        print("[{}/{}] train/test samples".format(n_tr_samples, n_te_samples))
        
        ################ 
        # fit on train #
        ################      
        preprocessor = Preprocessing()
        # vectorizer = CharVectorizer(maxlen=opt.maxlen, padding='post', truncating='post')
        vectorizer = ByteVectorizer(maxlen=opt.maxlen, padding='post', truncating='post', dataset=opt.dataset)
        for sentence, label in tqdm(dataset.load_train_data(), desc="fit on train...", total= n_tr_samples):
            s_prepro = preprocessor.transform([sentence])
            vectorizer.partial_fit(s_prepro)

        del tr_sentences
        del te_sentences

        ###################
        # transform train #
        ###################
        with lmdb.open(variables['train']['path'], map_size=1099511627776) as env:
            with env.begin(write=True) as txn:
                for i, (sentence, label) in enumerate(tqdm(dataset.load_train_data(), desc="transform train...", total= n_tr_samples)):

                    xtxt = vectorizer.transform(preprocessor.transform([sentence]))[0]
                    lab = label

                    txt_key = 'txt-%09d' % i
                    lab_key = 'lab-%09d' % i
                    
                    txn.put(lab_key.encode(), list_to_bytes([lab]))
                    txn.put(txt_key.encode(), list_to_bytes(xtxt))

                txn.put('nsamples'.encode(), list_to_bytes([i+1]))

        ##################
        # transform test #
        ##################
        with lmdb.open(variables['test']['path'], map_size=1099511627776) as env:
            with env.begin(write=True) as txn:
                for i, (sentence, label) in enumerate(tqdm(dataset.load_test_data(), desc="transform test...", total= n_te_samples)):

                    xtxt = vectorizer.transform(preprocessor.transform([sentence]))[0]
                    lab = label

                    txt_key = 'txt-%09d' % i
                    lab_key = 'lab-%09d' % i
                    
                    txn.put(lab_key.encode(), list_to_bytes([lab]))
                    txn.put(txt_key.encode(), list_to_bytes(xtxt))

                txn.put('nsamples'.encode(), list_to_bytes([i+1]))

        variables['txt_dict']['var'] = vectorizer.get_params()
        n_tokens = len(variables['txt_dict']['var']['char_dict'])

        ###############
        # saving data #
        ###############     
        print("  - saving to {}".format(variables['txt_dict']['path']))
        pkl.dump(variables['txt_dict']['var'],open(variables['txt_dict']['path'],"wb"))
        
    tr_loader = DataLoader(TupleLoader(variables['train']['path']), batch_size=opt.batch_size, shuffle=False, num_workers=opt.nthreads, pin_memory=True)
    te_loader = DataLoader(TupleLoader(variables['test']['path']), batch_size=opt.batch_size, shuffle=False, num_workers=opt.nthreads, pin_memory=False)

    # select cpu or gpu
    device = torch.device("cuda:{}".format(opt.gpuid) if opt.gpuid >= 0 else "cpu")
    list_metrics = ['accuracy']

    net = SVDCNN(n_classes=n_classes, num_embedding=n_tokens, embedding_dim=16, depth=opt.depth, shortcut=True)

    print("\nCreating model...")
    criterion = torch.nn.CrossEntropyLoss()
    net.to(device)

    total_param, total_param_conv, total_param_fc = count_parameters(net)
    print('\nNum of parameters: ')
    print('#conv (M): %0.2f' % (total_param_conv/pow(10,6)))
    print('#fc (M): %0.2f' % (total_param_fc/pow(10,6)))
    print("#total (M): %0.2f" % (total_param/pow(10,6)))
    print("Size (MB): %0.2f\n" % abs(total_param * 4. / (1024 ** 2.)))

    print(" - optimizer: sgd\n")
    optimizer = torch.optim.SGD(net.parameters(), lr = opt.lr, momentum=0.9, weight_decay=0.001)

    scheduler = None
    if opt.lr_halve_interval > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_halve_interval, gamma=opt.gamma, last_epoch=-1)
        
    for epoch in range(1, opt.epochs + 1):
        train(epoch,net, tr_loader, device, msg="training", optimize=True, optimizer=optimizer, scheduler=scheduler, criterion=criterion)
        train(epoch,net, te_loader, device, msg="testing ", criterion=criterion)

        if (epoch % opt.snapshot_interval == 0) and (epoch > 0):
            path = "{}/model_epoch_{}".format(opt.model_folder,epoch)
            print("snapshot of model saved as {}".format(path))
            save(net, variables['txt_dict']['var'], path=path)

    if opt.epochs > 0:
        path = "{}/model_epoch_{}".format(opt.model_folder,opt.epochs)
        print("snapshot of model saved as {}".format(path))
        save(net, variables['txt_dict']['var'], path=path)

    print('\n\n\n\n\n')