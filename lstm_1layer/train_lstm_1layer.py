import os
import numpy as np
import pickle
import json
import sys

abspath = os.path.abspath(__file__)
filename = os.sep.join(abspath.split(os.sep)[-2:])
abspath = abspath.replace(filename, "")
sys.path.append(abspath)

from net.LSTMCell import lstmcell_layer
from net.fullconnect import fclayer
from net.loss import cross_entropy_loss

def train():
    inputs_character = "abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz "
    unique = set(list(inputs_character))
    length = len(unique)
    id2char = {i:char for i, char in enumerate(unique)}
    char2id = {char:i for i, char in enumerate(unique)}

    jsonpth = os.path.join(abspath, 'dataset', r"lstm1layer.json")
    if not os.path.exists(jsonpth):
        with open(jsonpth, 'w') as obj:
            json.dump({"id2char":id2char, 'char2id':char2id}, obj, indent=2, separators=(",", ":"))
    else:
        with open(jsonpth, 'r') as obj:
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
    batch_size = 1
    sequence_length = 10
    bias = True

    epoch = 2000 * 9
    lr = 0.001
    lstmnet = lstmcell_layer(input_size, hidden_size, bias)
    fullconnect = fclayer(hidden_size, input_size, True)

    for e in range(epoch):
        hidden_input =[]
        hidden_col_fc =[]
        loss_col = []
        delta_col = []
        predict_col = []
        inputs_col = []
        middle_output_col = []
        hidden_0 = np.zeros((batch_size, hidden_size))
        cell_0 = np.zeros((batch_size, hidden_size))
        hidden_delta = np.zeros((batch_size, hidden_size))
        cell_delta = np.zeros((batch_size, hidden_size))

        if epoch ==1600*9:
            lr *= 0.1
        elif epoch==1800*9:
            lr *=0.1
        inputs = np.zeros((sequence_length, batch_size, input_size))
        labels = np.zeros((sequence_length, batch_size, input_size))
        for i in range(batch_size):
            choose = np.random.randint(len(inputs_character) - sequence_length - 1)
            inp = [char2id[i] for i in inputs_character[choose : choose + sequence_length]]
            oup = [char2id[i] for i in inputs_character[choose + 1 : choose + sequence_length+1]]
            charin = ''.join([id2char[i] for i in inp])
            charou = ''.join([id2char[i] for i in oup])
            inputs[np.arange(sequence_length), i, inp] = 1
            labels[np.arange(sequence_length), i, oup] = 1

        for j in range(sequence_length):
            hidden_input.append((hidden_0, cell_0))

            hidden_cell_next, middle_output = lstmnet.forward(inputs[j, :, :], (hidden_0, cell_0))
            hidden_0, cell_0 = hidden_cell_next

            hidden_col_fc.append(hidden_0)
            middle_output_col.append(middle_output)

            y = fullconnect.forward(hidden_0)

            loss, delta, predict = cross_entropy_loss(y, labels[j, :, :])
            inputs_col.append(inputs[j, :, :])
            loss_col.append(loss)
            delta_col.append(delta)
            predict_col.append(predict)

        for j in np.arange(sequence_length-1, -1, -1):
            d_h_i1 = fullconnect.backward(delta_col[j], hidden_col_fc[j])
            _i_, _f_, _g_, _o_, c_tanhk = middle_output_col[j]
            input_delta, hidden_delta, cell_delta = lstmnet.backward(d_h_i1, hidden_delta, cell_delta, inputs_col[j], hidden_input[j], _i_, _f_, _g_, _o_, c_tanhk)

        lstmnet.update(lr)
        fullconnect.update(lr)
        lstmnet.setzero()
        fullconnect.setzero()
        meanloss = np.mean(loss_col)

        hidden_0_ = np.zeros((1, hidden_size))
        cell_0_ = np.zeros((1, hidden_size))
        result = ''
        first_inputs = inputs[0, -1, :][np.newaxis, :]
        for j in range(sequence_length):
            (hidden_0_, cell_0_), middle_output = lstmnet.forward(first_inputs, (hidden_0_, cell_0_))
            y = fullconnect.forward(hidden_0_)
            p_shift = y - np.max(y, axis = -1)[:, np.newaxis]   # avoid too large in exp
            predict = np.exp(p_shift) / np.sum(np.exp(p_shift), axis = -1)[:, np.newaxis]
            p = np.argmax(predict, axis=-1)[0]
            first_inputs  = np.zeros((1, input_size))
            first_inputs[0, p] = 1
            rek = id2char[p]
            result += rek
        print(e, meanloss, charin, charou, result)

    model = [lstmnet.save_model(), fullconnect.save_model()]
    with open(os.path.join(abspath, 'models', 'lstm1layer.pkl'), 'wb') as obj:
        pickle.dump(model, obj)

if __name__=="__main__":
    train()