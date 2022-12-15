import os
import csv
import json
import lmdb
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import torch.nn.functional as F
from sklearn import utils, metrics
from torch.utils.data import DataLoader, Dataset
from src.main import TupleLoader, list_from_bytes, get_metrics, count_parameters, Preprocessing, ByteVectorizer, \
    list_to_bytes
from src.net import SVDCNN
from src.datasets import load_datasets


class text_loader:
    def __init__(self, sentences):
        self.left_test, self.right_test = self.read_sentences(sentences)

    def read_sentences(self, pairs):
        l_sentences = []
        r_sentences = []
        for pair in pairs:
            left_line = pair[0].decode("latin-1")
            # line = re.sub(r"(?:\\x[A-Fa-f0-9]{2})", "", line)
            left_line = left_line.replace("\\n", "")
            left_line = left_line[0:-1]
            right_line = pair[1].decode("latin-1")
            right_line = right_line.replace("\\n", "")
            right_line = right_line[0:-1]
            if left_line != right_line:
                l_sentences.append(left_line)
                r_sentences.append(right_line)
        return l_sentences, r_sentences

    def get_data(self):
        return self.left_test, self.right_test


def bytewise(sentences):
    data = text_loader(sentences)
    left_test, right_test = data.get_data()
    test_byte_rows_str = []
    test_byte_rows_list = []
    test_ltb = []
    test = []
    for idx, (l1, l2) in tqdm(enumerate(zip(left_test, right_test))):
        new_l1_str = ""
        new_l1_list = []
        new_l2_str = " "
        new_l2_list = []
        for c1, c2 in zip(l1, l2):
            with open('./datasets/demo/raw/char2byte.pkl', 'rb') as f:
                char2byte = pkl.load(f)
            new_l1_str = new_l1_str + str(char2byte[c1]) + " "
            new_l2_str = new_l2_str + str(char2byte[c2]) + " "
            new_l1_list.append(char2byte[c1])
            new_l2_list.append(char2byte[c2])
        if new_l1_str != new_l2_str:
            # byte_rows_str.append([new_l1_str, new_l1_str, new_l2_str])
            test_ltb.append(['1', '', f'{new_l1_str}'])
            test_ltb.append(['2', '', f'{new_l2_str}'])
            temp1 = []
            temp2 = []
            l1_str = ""
            l2_str = ""
            meet = False
            curr = -1
            for i, (byte1, byte2) in enumerate(zip(new_l1_list, new_l2_list)):
                if byte1 != byte2:
                    if not meet:
                        # print(i)
                        # print(max(0, i-5))
                        temp1.extend(new_l1_list[max(0, i - 5):i + 1])
                        temp2.extend(new_l2_list[max(0, i - 5):i + 1])
                        meet = True
                        curr = i + 1
                    else:
                        temp1.extend(new_l1_list[max(curr, i - 5):i + 1])
                        temp2.extend(new_l2_list[max(curr, i - 5):i + 1])
                        curr = i + 1
            temp1.extend(new_l1_list[curr:min(curr + 5, len(new_l1_list))])
            temp2.extend(new_l2_list[curr:min(curr + 5, len(new_l2_list))])
            for byte1 in temp1:
                l1_str += str(byte1) + " "
            for byte2 in temp2:
                l2_str += str(byte2) + " "
            test.append(['1', '', f'{l1_str}'])
            test.append(['2', '', f'{l2_str}'])
            # test_byte_row_json = json.dumps({'id': idx, 'en': new_l1_list, 'unknown': new_l2_list})
    if os.path.exists("./datasets/demo/raw/demo.csv"):
        os.remove("./datasets/demo/raw/demo.csv")
    with open('./datasets/demo/raw/demo.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        # writer.writerow(header)
        for ltb in tqdm(test):
            writer.writerow(ltb)
    return "./datasets/demo/raw/demo.csv"


def cacc(y_probs):
    yp = np.concatenate(y_probs, 0)
    preds = []
    for i in range(0, len(yp), 2):
        if yp[i][0] >= yp[i][1]:
            pred1 = 0
        else:
            pred1 = 1
        if yp[i + 1][0] >= yp[i + 1][1]:
            pred2 = 0
        else:
            pred2 = 1
        if pred1 != pred2:
            preds.append(pred1)
        else:
            if yp[i][0] >= yp[i + 1][0]:
                pred1 = 0
            else:
                pred1 = 1
            preds.append(pred1)
    return preds


def predict(net, dataset, device, msg="prediction"):
    net.eval()
    y_probs, y_trues = [], []
    for iteration, (tx, ty) in tqdm(enumerate(dataset), total=len(dataset), desc="{}".format(msg)):
        data = (tx, ty)
        data = [x.to(device) for x in data]
        out = net(data[0])
        ty_prob = F.softmax(out, 1)  # probabilites
        y_probs.append(ty_prob.detach().cpu().numpy())
        y_trues.append(data[1].detach().cpu().numpy())
    preds = cacc(y_probs)

    preds_message = []
    for digit in preds:
        if digit == 0:
            preds_message.append("The 1st sentence is English.")
        elif digit == 1:
            preds_message.append("The 2rd sentence is English.")
    return preds_message


if __name__ == "__main__":
    sentences = [
        [b"On the way , Officer Satou and Takagi make a detour to Beika Woods after receiving a call from Professor Agasa .",
            b"On the way , Officer Satou and Takagi make a detour to Beika Woods after receiving a from Professor Agasa ."],
        [b"Shattered , she returns back and gets married to the Captain .",
         b"Shattered , returns back and gets married to the Captain ."],
        [b"Reconciling family and working life and gender mainstreaming are not matters for women alone.",
         b"Reconciling family and working life and gender mainstreaming not matters for women alone."],
        [b"This makes him somewhat recognizable to Tamiko .",
         b"This makes him somewhat recognizable to Twippo ."],
        [b"The President 's science aides set up a first contact meeting with the Martians in Pahrump , Nevada .",
         b"President The 's science aides set up a first contact meeting with the Martians in Pahrump , Nevada ."]
    ]

    # sentences = [[b"En route , they pick up a seemingly-harmless hitchhiker , and continue their journey , only for their car to break down in a deserted motel on a lonely highway .",
    #             b"En route , they pick up a seemingly-harmless hitchhiker , and continue their , journey only for their car to break down in a deserted motel on a lonely highway ."],
    #              [b"I should just like to highlight two issues.", b"I should just like to hihglight two issues."],
    #              [b"Jerry is finally released from hospital and returns to London to look for Diana .", b"Jerry is finally released from hospital and returns London to look for Diana ."]]
    demo_path = bytewise(sentences)
    epoch_path = './models/noah_17_0.05/model_epoch_50'
    checkpoint = torch.load(epoch_path)
    n_classes = 2
    depth = 17
    dataset = "demo"
    data_folder = "datasets"
    data_folder = f"{data_folder}/{dataset}/svdcnn_17_0.05"
    os.makedirs(data_folder, exist_ok=True)
    dataset = load_datasets(names=[dataset])[0]
    dataset_name = dataset.__class__.__name__
    n_classes = dataset.n_classes
    print("dataset: {}, n_classes: {}".format(dataset_name, n_classes))

    device = torch.device("cuda:{}".format(0) if 0 >= 0 else "cpu")
    list_metrics = ['accuracy']
    variables = {
        'test': {'var': None, 'path': "{}/test.lmdb".format(data_folder)},
        'txt_dict': {'var': None, 'path': "{}/txt_dict.pkl".format(data_folder)},
    }


    print("Creating datasets")
    demo_sentences = [txt for txt, lab in tqdm(dataset.load_test_data(), desc="counting test samples")]
    n_demo_samples = len(demo_sentences)

    print("[{}] demo samples".format(n_demo_samples))
    preprocessor = Preprocessing()
    vectorizer = ByteVectorizer(maxlen=1024, padding='post', truncating='post', dataset="demo")
    ##################
    # transform demo #
    ##################
    with lmdb.open(variables['test']['path'], map_size=1099511627776) as env:
        with env.begin(write=True) as txn:
            for i, (sentence, label) in enumerate(
                    tqdm(dataset.load_test_data(), desc="transform demo...", total=n_demo_samples)):
                xtxt = vectorizer.transform(preprocessor.transform([sentence]))[0]
                lab = label

                txt_key = 'txt-%09d' % i
                lab_key = 'lab-%09d' % i

                txn.put(lab_key.encode(), list_to_bytes([lab]))
                txn.put(txt_key.encode(), list_to_bytes(xtxt))

            txn.put('nsamples'.encode(), list_to_bytes([i + 1]))

    variables['txt_dict']['var'] = vectorizer.get_params()
    n_tokens = len(variables['txt_dict']['var']['char_dict'])

    ###############
    # saving data #
    ###############
    print("  - saving to {}".format(variables['txt_dict']['path']))
    pkl.dump(variables['txt_dict']['var'], open(variables['txt_dict']['path'], "wb"))

    model = SVDCNN(n_classes=n_classes, num_embedding=n_tokens, embedding_dim=16, depth=depth, shortcut=True)
    del checkpoint['txt_dict']
    model.load_state_dict(checkpoint)
    model.to(device)
    te_loader = DataLoader(TupleLoader(variables['test']['path']), batch_size=1, shuffle=False, num_workers=4,
                           pin_memory=False)

    total_param, total_param_conv, total_param_fc = count_parameters(model)
    print('\nNum of parameters: ')
    print('#conv (M): %0.2f' % (total_param_conv / pow(10, 6)))
    print('#fc (M): %0.2f' % (total_param_fc / pow(10, 6)))
    print("#total (M): %0.2f" % (total_param / pow(10, 6)))
    print("Size (MB): %0.2f\n" % abs(total_param * 4. / (1024 ** 2.)))
    print(" - optimizer: sgd\n")

    preds_message = predict(model, te_loader, device)
    for m in preds_message:
        print(m)
