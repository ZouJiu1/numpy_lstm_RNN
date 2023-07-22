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
from net.RNNCell import rnncell_layer
from net.fullconnect import fclayer
from net.Embedding import Embedding_layer

def predict():
    num_layer = 3
    with open(os.path.join(abspath, 'dataset', r"tangshi_rnn2layer_dynamic_highfreq.json"), 'r') as obj:
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
    num_layer = len(hidden_size)
    bias = True

    embedding = Embedding_layer(length, embedding_dim = embedding_dim)
    rnn_layer0 = rnncell_layer(embedding_dim, hidden_size[0], bias)
    rnn_layer1 = rnncell_layer(hidden_size[0], hidden_size[1], bias)
    rnnlayers = [rnn_layer0, rnn_layer1]
    
    rnn_decode0 = rnncell_layer(embedding_dim, hidden_size[0], bias)
    rnn_decode1 = rnncell_layer(hidden_size[0], hidden_size[1], bias)
    rnn_decode = [rnn_decode0, rnn_decode1]
    fullconnect = fclayer(hidden_size[1], length, True)

    if os.path.exists(pth):
        with open(pth, 'rb') as obj:
            models = pickle.load(obj)
    else:
        exit(-1)

    embedding.restore_model(models[0])
    rnnlayers[0].restore_model(models[1])
    rnnlayers[1].restore_model(models[2])
    rnn_decode[0].restore_model(models[3])
    rnn_decode[1].restore_model(models[2*2])
    fullconnect.restore_model(models[2*2+1])
    start_iters = models[-1]

    result = ""
    batch_size = 1
    start = "拂局尽消时，"
    first_inputs = np.array([int(char2id[i]) for i in start])
    result += start
    
    length = 20
    output_length = 6 - 1
    hidden = []
    for ind in range(num_layer):
        hidden.append(np.zeros((batch_size, hidden_size[ind])))
    for j in range(length):
        hidden = []
        for ind in range(num_layer):
            hidden.append(np.zeros((1, hidden_size[ind])))
        for ij in range(len(first_inputs)):
            embedding_output = embedding.forward(first_inputs[ij])
            hidden[0] = rnnlayers[0].forward(embedding_output, hidden[0])
            hidden[1] = rnnlayers[1].forward(hidden[0], hidden[1])

        zerosin = np.zeros_like(embedding_output)
        for _ in range(output_length):
            hidden[0] = rnn_decode[0].forward(zerosin, hidden[0])
            hidden[1] = rnn_decode[1].forward(hidden[0], hidden[1])
                    
            y = fullconnect.forward(hidden[1])
            p_shift = y - np.max(y, axis = -1)[:, np.newaxis]   # avoid too large in exp
            predict = np.exp(p_shift) / np.sum(np.exp(p_shift), axis = -1)[:, np.newaxis]
            p = np.argmax(predict, axis=-1)[0]
            rek = id2char[p]
            result += rek
    print(result)

if __name__=="__main__":
    pth = os.path.join(abspath, "models", 'tangshi_rnn2layer_embedding_dynamic.pkl')
    predict()