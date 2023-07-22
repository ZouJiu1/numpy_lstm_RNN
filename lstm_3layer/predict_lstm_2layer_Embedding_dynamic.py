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
    num_layer = len(hidden_size)
    bias = True

    embedding = Embedding_layer(length, embedding_dim = embedding_dim)
    lstm_layer0 = lstmcell_layer(embedding_dim, hidden_size[0], bias)
    lstm_layer1 = lstmcell_layer(hidden_size[0], hidden_size[1], bias)
    lstmlayers = [lstm_layer0, lstm_layer1]
    
    lstm_decode0 = lstmcell_layer(embedding_dim, hidden_size[0], bias)
    lstm_decode1 = lstmcell_layer(hidden_size[0], hidden_size[1], bias)
    lstm_decode = [lstm_decode0, lstm_decode1]
    fullconnect = fclayer(hidden_size[1], length, True)

    if os.path.exists(pth):
        with open(pth, 'rb') as obj:
            models = pickle.load(obj)
    else:
        exit(-1)

    embedding.restore_model(models[0])
    lstmlayers[0].restore_model(models[1])
    lstmlayers[1].restore_model(models[2])
    lstm_decode[0].restore_model(models[3])
    lstm_decode[1].restore_model(models[2*2])
    fullconnect.restore_model(models[2*2+1])
    start_iters = models[-1]

    result = ""
    batch_size = 1
    #["月","在","画","楼","西"] ["楼","高","秋","易","寒"] ["更","上","最","高","楼"] ["愁","倚","溪","楼","望"] ["朝","送","山","僧","去"] ['寒', "向", '江南', '暖', "日"]  ["携", '杖', '溪', '边', "听"] ["残", "星", "落", "檐", "外"] ["暮","景","千","山","雪"]
    start = ["楼","高","秋","易","寒"]
    first_inputs = np.array([int(char2id[i]) for i in start])
    
    length = 2 + 1
    output_length = 6 - 1
    hidden = []
    cell = []
    for ind in range(num_layer):
        hidden.append(np.zeros((batch_size, hidden_size[ind])))
        cell.append(np.zeros((1, hidden_size[ind])))
    rekkk = "".join(start) + ",\n"
    for j in range(length):
        hidden = []
        for ind in range(num_layer):
            hidden.append(np.zeros((1, hidden_size[ind])))
        for ij in range(len(first_inputs)):
            embedding_output = embedding.forward(first_inputs[ij])
            (hidden[0], cell[0]), middle_output = lstmlayers[0].forward(embedding_output, (hidden[0], cell[0]))
            (hidden[1], cell[1]), middle_output = lstmlayers[1].forward(hidden[0], (hidden[1], cell[1]))

        zerosin = np.zeros_like(embedding_output)
        result = ""
        first_inputs = []
        for _ in range(output_length):
            (hidden[0], cell[0]), middle_output = lstm_decode[0].forward(zerosin, (hidden[0], cell[0]))
            (hidden[1], cell[1]), middle_output = lstm_decode[1].forward(hidden[0], (hidden[1], cell[1]))

            y = fullconnect.forward(hidden[1])
            p_shift = y - np.max(y, axis = -1)[:, np.newaxis]   # avoid too large in exp
            predict = np.exp(p_shift) / np.sum(np.exp(p_shift), axis = -1)[:, np.newaxis]
            p = np.argmax(predict, axis=-1)[0]
            first_inputs.append(p)
            rek = id2char[p]
            result += rek
        rekkk += result +",\n"
    print(rekkk)

if __name__=="__main__":
    frequency = 1000
    pth = os.path.join(abspath, "models", 'tangshi_lstm2layer_embedding_dynamic_%d.pkl'%frequency)
    predict()