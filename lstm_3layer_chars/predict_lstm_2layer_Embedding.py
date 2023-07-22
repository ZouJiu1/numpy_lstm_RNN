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

def predict():
    num_layer = 3
    with open(os.path.join(abspath, 'dataset', r"tangshi_lstm2layer.json"), 'r') as obj:
        jsonfile = json.load(obj)
    id2chark = jsonfile["id2char"]
    char2id = jsonfile["char2id"]
    length = len(id2chark)
    id2char = {}
    for key, value in id2chark.items():
        id2char[int(key)] = value
    end = '&'
    endid = char2id['&']

    embedding_dim  = 100
    hidden_size = [300, 300]
    sequence_length = 6
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
    start = "拂局尽消时，"
    first_inputs = np.zeros((sequence_length, 1), dtype=np.uint32)
    first_inputs[:, 0] = [int(char2id[i]) for i in start]
    result += start
    
    length = 20
    hidden = []
    cell = []
    for ind in range(num_layer):
        hidden.append(np.zeros((batch_size, hidden_size[ind])))
        cell.append(np.zeros((1, hidden_size[ind])))
    for j in range(length):
        for i in range(sequence_length):
            inputs = np.array([first_inputs[i, :]])
            embedding_output = embedding.forward(inputs)
            (hidden[0], cell[0]), middle_output = lstmlayers[0].forward(embedding_output, (hidden[0], cell[0]))
            (hidden[1], cell[1]), middle_output = lstmlayers[1].forward(hidden[0], (hidden[1], cell[1]))
            if i==sequence_length-1:
                y = fullconnect.forward(hidden[1])
                p_shift = y - np.max(y, axis = -1)[:, np.newaxis]                    # avoid too large in exp
                predict = np.exp(p_shift) / np.sum(np.exp(p_shift), axis = -1)[:, np.newaxis]
                p = np.argmax(predict, axis=-1)[0]
                rek = id2char[int(p)]
                if rek==end:
                    result += "。"
                    break
                result += rek
                first_inputs = np.zeros((sequence_length, 1), dtype=np.uint32)
                first_inputs[:, 0] = [int(char2id[i]) for i in result[-sequence_length:]]
    print(result)

if __name__=="__main__":
    pth = os.path.join(abspath, "models", 'tangshi_lstm2layer_embedding.pkl')
    predict()