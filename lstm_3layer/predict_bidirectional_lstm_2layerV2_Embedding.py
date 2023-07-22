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
    with open(inpath, 'r', encoding='utf-%d' % (2*2*2)) as obj:
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
    num_layer = len(hidden_size)
    bias = True

    embedding = Embedding_layer(length, embedding_dim = embedding_dim)
    lstm_layer0 = lstmcell_layer(embedding_dim, hidden_size[0], bias)
    lstm_layer1 = lstmcell_layer(hidden_size[0], hidden_size[1], bias)
    lstmlayers = [lstm_layer0, lstm_layer1]
    fullconnect = fclayer(hidden_size[1] * 2, length, True)

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
    #['寒', "向", '江南', '暖', "日"]  ["携", '杖', '溪', '边', "听"] ["残", "星", "落", "檐", "外"] ["暮","景","千","山","雪"]
    start = ["残", "星", "落", "檐", "外"]
    sequence_length = max(6-1, len(start))
    first_inputs = np.zeros((sequence_length, 1), dtype=np.uint32)
    first_inputs[:, 0] = [int(char2id[i]) for i in start]
    result += start[0]
    
    length = 2*2
    hidden = []
    cell = []
    hidden_r = []
    cell_r = []
    for ind in range(num_layer):
        hidden.append(np.zeros((batch_size, hidden_size[ind])))
        cell.append(np.zeros((1, hidden_size[ind])))
        hidden_r.append(np.zeros((batch_size, hidden_size[ind])))
        cell_r.append(np.zeros((1, hidden_size[ind])))
    for j in range(length):
        hidden_col_in = []
        hidden_col_in_r = []
        for i in range(sequence_length):
            # if j==0:
            inputs = first_inputs[i, :]
            inputs = np.array([inputs])
            embedding_output = embedding.forward(inputs)
            (hidden[0], cell[0]), middle_output = lstmlayers[0].forward(embedding_output, (hidden[0], cell[0]))
            (hidden[1], cell[1]), middle_output = lstmlayers[1].forward(hidden[0], (hidden[1], cell[1]))
            hidden_col_in.append(hidden[1][0])

            r_i = sequence_length - i - 1
            embedding_output = embedding.forward(first_inputs[r_i, :])
            (hidden_r[0], cell_r[0]), middle_in_col_0 = lstmlayers[0].forward(embedding_output, (hidden_r[0], cell_r[0]))
            (hidden_r[1], cell_r[1]), middle_in_col_1 = lstmlayers[1].forward(hidden_r[0], (hidden_r[1], cell_r[1]))
            hidden_col_in_r.append(hidden_r[1])

        for j in range(sequence_length):
            hidden_in = np.concatenate([hidden_col_in[j], hidden_col_in_r[sequence_length - j - 1]], axis = 1)
            y = fullconnect.forward(hidden_in)
            p_shift = y - np.max(y, axis = -1)[:, np.newaxis]                    # avoid too large in exp
            predict = np.exp(p_shift) / np.sum(np.exp(p_shift), axis = -1)[:, np.newaxis]
            p = np.argmax(predict, axis=-1)[0]
            first_inputs[j, 0] = p
            rek = id2char[int(p)]
            # if rek!=end and rek!="。" and rek!="？" and rek!="，":
            #     inputs = p
            if rek==end:
                break
            result += rek
    kp = ""
    for i in range(len(result)-1):
        if (i+1)%len(start)==0:
            if i%2==0:
                kp += result[i] + "，\n"
            else:
                kp += result[i] + "。\n"
        else:
            kp += result[i]
    print(kp)

if __name__=="__main__":
    frequency = 1000
    pth = os.path.join(abspath, "models", 'tangshi_bidirectional_lstm2layerV2_embedding_%d.pkl'%frequency)
    predict()