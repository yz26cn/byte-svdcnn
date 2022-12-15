import os
import json
import lmdb
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import torch.nn.functional as F
from sklearn import utils, metrics
from torch.utils.data import DataLoader, Dataset
from src.main import TupleLoader, list_from_bytes, get_metrics, count_parameters
from src.net import SVDCNN

def cacc(y_probs, y_trues):
    yp = np.concatenate(y_probs, 0)
    yt = np.concatenate(y_trues, 0).reshape(-1, 1)
    total_comp = 0
    true_comp = 0
    total = 0
    true = 0
    for i in range(0, len(yp), 2):
        if yt[i][0] == 0 and yt[i + 1][0] == 1 and yp[i][0] > yp[i + 1][0]:
            true_comp += 1
        total_comp += 1
    for i in range(len(yp)):
        if yp[i][0] >= yp[i][1]:
            if yt[i][0] == 0:
                true += 1
        else:
            if yt[i][0] == 1:
                true += 1
        total += 1
    return true_comp / total_comp, true / total

def predict(net,dataset,device,msg="prediction"):
    net.eval()
    y_probs, y_trues = [], []
    for iteration, (tx, ty) in tqdm(enumerate(dataset), total=len(dataset), desc="{}".format(msg)):
        data = (tx, ty)
        data = [x.to(device) for x in data]
        out = net(data[0])
        ty_prob = F.softmax(out, 1) # probabilites
        y_probs.append(ty_prob.detach().cpu().numpy())
        y_trues.append(data[1].detach().cpu().numpy())
        # if iteration % 999 == 0 and iteration != 0:
        #     print(iteration, ": ", cacc(y_probs, y_trues))
    acc_comp, acc_true = cacc(y_probs, y_trues)
    return acc_comp, acc_true

if __name__ == "__main__":
    sentences = [
        r'I am absolutely convinced that if this body had existed from the start, we would have managed to be better coordinated, we would have known more, and we would have been more effective.	I am absolutely convinced that if this body had existed from the start, we would have managed to be better celebration, we would have known more, and we would have been more effective.',
        r'The competent Greek authorities have informed me of the election of Mr Dimitrios Droutsas to replace Mr Lambrinidis.	The competent Greek authorities have informed me of the election Mr Dimitrios Droutsas to replace Mr Lambrinidis. ',
        r"This proves difficult , and is made more so by other incidents including attempts to kill him , a series of gruelling tasks set by the Wachootoo , and Wachati princess 's attempts to seduce him . 	This proves difficult , and is made more so by other incidents including attempts to kill him , a series of gruelling tasks set by the Wachootoo , and the Wachati princess 's attempts to seduce him .",
        ]
    epoch_path = './models/noah_17_0.05/model_epoch_50'
    checkpoint = torch.load(epoch_path)
    n_classes = 2
    depth = 17
    dataset = "noah"
    data_folder = "datasets"
    data_folder = f"{data_folder}/{dataset}/svdcnn_0.05"
    device = torch.device("cuda:{}".format(0) if 0 >= 0 else "cpu")
    list_metrics = ['accuracy']
    variables = {
        'test': {'var': None, 'path': "{}/test.lmdb".format(data_folder)},
        'txt_dict': {'var': None, 'path': "{}/txt_dict.pkl".format(data_folder)},
    }

    all_exist = True if os.path.exists(variables['txt_dict']['path']) else False
    if all_exist:
        # print("  - Loading: {}".format(variables['txt_dict']['path']))
        variables['txt_dict']['var'] = pkl.load(open(variables['txt_dict']['path'], "rb"))
        n_tokens = len(variables['txt_dict']['var']['char_dict'])
    else:
        print("not working")
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

    acc_comp, acc_true = predict(model, te_loader, device)
    print(acc_comp, acc_true)