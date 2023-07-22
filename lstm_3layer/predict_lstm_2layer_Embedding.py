import numpy as np
import sys
import os
import pickle
import json

np.set_printoptions(suppress=True)
abspath = os.path.abspath(__file__)
filename = os.sep.join(abspath.split(os.sep)[-2:])
abspath = abspath.replace(filename, "")

sys.path.append(abspath)
from net.LSTMCell import lstmcell_layer
from net.fullconnect import fclayer
from net.Embedding import Embedding_layer
from tokenlzh import EOS

def predict():
    num_layer = 3
    inpath = os.path.join(abspath, 'dataset', r"id2char_char2id_%d.json"%frequency)
    with open(inpath, 'r', encoding='utf-%d'%(2*2*2)) as obj:
        jsonfile = json.load(obj)
    id2chark = jsonfile["id2char"]
    char2id = jsonfile["char2id"]
    length = len(id2chark)
    id2char = {}
    for key, value in id2chark.items():
        id2char[int(key)] = value
    end = EOS
    endid = char2id[EOS]

    embedding_dim  = 100
    hidden_size = [300, 300]
    sequence_length = 6 - 1
    num_layer = len(hidden_size)
    bias = True

    embedding = Embedding_layer(length, embedding_dim = embedding_dim)
    lstm_layer0 = lstmcell_layer(embedding_dim, hidden_size[0], bias)
    lstm_layer1 = lstmcell_layer(hidden_size[0], hidden_size[1], bias)
    lstmlayers = [lstm_layer0, lstm_layer1]
    fullconnect = fclayer(hidden_size[1], length, True)

    if os.path.exists(pth):
        with open(pth, 'rb') as obj:
            models = pickle.load(obj)
    else:
        exit(-1)
    embedding.restore_model(models[0])
    lstm_layer0.restore_model(models[1])
    lstm_layer1.restore_model(models[2])
    fullconnect.restore_model(models[3])

    result = ""
    batch_size = 1
    #["朝","送","山","僧","去"] ['寒', "向", '江南', '暖', "日"]  ["携", '杖', '溪', '边', "听"] ["残", "星", "落", "檐", "外"] ["暮","景","千","山","雪"]
    start = ["朝","送","山","僧","去"] #"寒向江南暖" #"携杖溪边听"
    assert len(start) == sequence_length
    first_inputs = np.zeros((sequence_length, 1), dtype=np.uint32)
    first_inputs[:, 0] = [int(char2id[i]) for i in start]
    result += "".join(start)
    
    length = 20-3-1
    hidden = []
    cell = []
    for ind in range(num_layer):
        hidden.append(np.zeros((batch_size, hidden_size[ind])))
        cell.append(np.zeros((1, hidden_size[ind])))
    tmp = []
    for j in range(length):
        for i in range(sequence_length):
            tmp.append(first_inputs[i, :])
            inputs = np.array([first_inputs[i, :]])
            embedding_output = embedding.forward(inputs)
            (hidden[0], cell[0]), middle_output = lstmlayers[0].forward(embedding_output, (hidden[0], cell[0]))
            (hidden[1], cell[1]), middle_output = lstmlayers[1].forward(hidden[0], (hidden[1], cell[1]))
            if i==sequence_length-1:
                y = fullconnect.forward(hidden[1])
                p_shift = y - np.max(y, axis = -1)[:, np.newaxis]                    # avoid too large in exp
                predict = np.exp(p_shift) / np.sum(np.exp(p_shift), axis = -1)[:, np.newaxis]
                p = np.argmax(predict, axis=-1)[0]
                tmp.append(p)
                rek = id2char[int(p)]
                if rek==end:
                    break
                result += rek
                first_inputs = np.zeros((sequence_length, 1), dtype=np.uint32)
                first_inputs[:, 0] = tmp[-sequence_length:]
    kp = ""
    for i in range(len(result)-1):
        if (i+1) % len(start)==0:
            if i % 2==0:
                kp += result[i] + "，\n"
            else:
                kp += result[i] + "。\n"
        else:
            kp += result[i]
    result = kp
    print(result)

if __name__=="__main__":
    frequency = 1000
    pth = os.path.join(abspath, "models", 'tangshi_lstm2layer_embedding_%d.pkl'%frequency)
    predict()