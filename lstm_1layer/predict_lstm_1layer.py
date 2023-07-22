import numpy as np
import sys
import os
import pickle
import json

abspath = os.path.abspath(__file__)
filename = os.sep.join(abspath.split(os.sep)[-2:])
abspath = abspath.replace(filename, "")
sys.path.append(abspath)

from net.LSTMCell import lstm_layer
from net.fullconnect import fclayer

def predict():
    with open(os.path.join(abspath, 'dataset', r"lstm1layer.json"), 'r') as obj:
        jsonfile = json.load(obj)
    id2chark = jsonfile["id2char"]
    char2id = jsonfile["char2id"]
    length = len(id2chark)
    id2char = {}
    for key, value in id2chark.items():
        id2char[int(key)] = value
    del id2chark
    for key, value in char2id.items():
        char2id[key] = int(value)

    input_size  = length
    hidden_size = 200
    sequence_length = 10
    bias = True
    
    lstmnet = lstm_layer(input_size, hidden_size, bias)
    fullconnect = fclayer(hidden_size, input_size, True)
    if os.path.exists(pth):
        with open(pth, 'rb') as obj:
            models = pickle.load(obj)
    else:
        exit(-1)
    lstmnet.restore_model(models[0])
    fullconnect.restore_model(models[1])

    result = ""
    batch_size = 1
    start = "a"
    first_inputs = np.zeros((1, input_size))
    first_inputs[0, char2id[start]] = 1
    result += start + " "
    hidden_0 = np.zeros((batch_size, hidden_size))
    cell_0 = np.zeros((batch_size, hidden_size))
    
    length = 20
    for j in range(length):
        (hidden_0, cell_0), middle_output = lstmnet.forward(first_inputs, (hidden_0, cell_0))
        y = fullconnect.forward(hidden_0)
        y = fullconnect.forward(hidden_0)
        p_shift = y - np.max(y, axis = -1)[:, np.newaxis]                    # avoid too large in exp
        predict = np.exp(p_shift) / np.sum(np.exp(p_shift), axis = -1)[:, np.newaxis]
        p = np.argmax(predict, axis=-1)[0]
        first_inputs  = np.zeros((batch_size, input_size))
        first_inputs[:, p] = 1
        rek = id2char[p]
        result += rek
    print(result)

if __name__=="__main__":
    pth = os.path.join(abspath, "models", 'lstm1layer.pkl')
    predict()