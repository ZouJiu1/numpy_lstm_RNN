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
    rnn_layer0 = rnncell_layer(embedding_dim, hidden_size[0], bias)
    rnn_layer1 = rnncell_layer(hidden_size[0], hidden_size[1], bias)
    rnnlayers = [rnn_layer0, rnn_layer1]
    fullconnect = fclayer(hidden_size[1], length, True)

    if os.path.exists(pth):
        with open(pth, 'rb') as obj:
            models = pickle.load(obj)
        embedding.restore_model(models[0])
        rnn_layer0.restore_model(models[1])
        rnn_layer1.restore_model(models[2])
        fullconnect.restore_model(models[3])
    else:
        # exit(-1)
        pass

    result = ""
    batch_size = 1
    #['寒', "向", '江南', '暖', "日"]  ["携", '杖', '溪', '边', "听"] ["残", "星", "落", "檐", "外"] ["暮","景","千","山","雪"]
    start = ["更","上","最","高","楼"]
    sequence_length = max(6-1, len(start))
    first_inputs = np.zeros((sequence_length, 1), dtype=np.uint32)
    first_inputs[:, 0] = [int(char2id[i]) for i in start]
    result += start[0]
    
    length = 2*2
    hidden = []
    for ind in range(num_layer):
        hidden.append(np.zeros((batch_size, hidden_size[ind])))
    for j in range(length):
        for i in range(sequence_length):
            # if j==0:
            inputs = first_inputs[i, :]
            inputs = np.array([inputs])
            embedding_output = embedding.forward(inputs)
            hidden[0] = rnnlayers[0].forward(embedding_output, hidden[0])
            hidden[1] = rnnlayers[1].forward(hidden[0], hidden[1])
            y = fullconnect.forward(hidden[1])
            p_shift = y - np.max(y, axis = -1)[:, np.newaxis]                    # avoid too large in exp
            predict = np.exp(p_shift) / np.sum(np.exp(p_shift), axis = -1)[:, np.newaxis]
            p = np.argmax(predict, axis=-1)[0]
            first_inputs[i, 0] = p
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
    pth = os.path.join(abspath, "models", 'tangshi_rnn2layrV2_embedding_%d.pkl'%frequency)
    predict()