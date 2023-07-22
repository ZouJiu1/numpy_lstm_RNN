import os
import numpy as np
import sys
import pickle
import json
np.set_printoptions(suppress=True)
abspath = os.path.abspath(__file__)
filename = os.sep.join(abspath.split(os.sep)[-2:])
abspath = abspath.replace(filename, "")

sys.path.append(abspath)

from net.LSTMCell import lstmcell_layer
from net.fullconnect import fclayer
from net.loss import cross_entropy_loss
from net.Embedding import Embedding_layer
from tokenlzh import preprocess

def train():
    id2char, char2id, length, all_lines, endid, end, quanzhong = \
        preprocess(frequency = frequency, delete_markchar = True)
    embedding_dim  = 100
    hidden_size = [300, 300]
    batch_size = 1
    sequence_length = 6 - 1
    num_layer = len(hidden_size)
    bias = True

    epoch = 100
    all_num = 0
    for i in all_lines:
        all_num += len(i)
    iters = (all_num // (sequence_length + 1) + 2*sequence_length) * epoch // batch_size
    showiter = 100
    savemodel = 1000
    learning_rate = 0.001
    
    embedding = Embedding_layer(length, embedding_dim = embedding_dim)
    lstm_layer0 = lstmcell_layer(embedding_dim, hidden_size[0], bias)
    lstm_layer1 = lstmcell_layer(hidden_size[0], hidden_size[1], bias)
    lstmlayers = [lstm_layer0, lstm_layer1]
    fullconnect = fclayer(hidden_size[1], length, True)
    
    # start_iters  = 0
    if os.path.exists(pretrained_model):
        with open(pretrained_model, 'rb') as obj:
            models = pickle.load(obj)
        embedding.restore_model(models[0])
        lstmlayers[0].restore_model(models[1])
        lstmlayers[1].restore_model(models[2])
        fullconnect.restore_model(models[3])
        start_iters = models[-1]
    else:
        # exit(-1)
        pass
    start_iters  = 0
    sentence = -6
    choose_lines = -1
    for e in range(iters):
        # hidden_col_lstm = [[] for i in range(num_layer)]
        # hidden_col_fc =[]
        hidden_lstm_in = [[] for i in range(num_layer)]
        hidden_col_in =[]
        loss_col = []
        delta_col = []
        predict_col = []
        # inputs_col = [[] for i in range(num_layer)]
        inputs_col = [[] for i in range(num_layer)]
        middle_in_col = [[] for i in range(num_layer)]
        hidden = []
        cell = []
        hidden_delta = []
        cell_delta   = []
        flatten_col = []
        for ind in range(num_layer):
            hidden.append(np.zeros((batch_size, hidden_size[ind])))
            cell.append(np.zeros((batch_size, hidden_size[ind])))
            hidden_delta.append(np.zeros((batch_size, hidden_size[ind])))
            cell_delta.append(np.zeros((batch_size, hidden_size[ind])))
        # if e < 1000:
        #     lr = 0.0001
        # else:
        lr = learning_rate

        if e == int(iters * (16/20)):
            lr = learning_rate * 0.1
        elif e == int(iters * (18/20)):
            lr = learning_rate * 0.1 * 0.1
        
        inputs = np.zeros((sequence_length, batch_size), dtype = np.int32)
        labels = np.zeros((batch_size, length), dtype = np.int32)
        for th in range(batch_size):
            if sentence < 0:
                choose_lines = (choose_lines + 1) % len(all_lines)
                inputs_character = all_lines[choose_lines]
                sentence = 9
                index = 0

            if len(inputs_character) - index < sequence_length + 1:
                sentence = -9
                inp = [i for i in inputs_character[index:-1]]
                inptwo = [endid for i in range(sequence_length - len(inp))]
                try:
                    oup = [inputs_character[-1]]
                except:
                    oup = [endid]
                index += len(inputs_character)
                charin = ''.join([id2char[i] for i in inp + inptwo])
                charou = ''.join([id2char[i] for i in oup])
                inputs[np.arange(len(inp)), th] = inp
                inputs[np.arange(len(inp), sequence_length), th] = inptwo
                labels[th, oup[0]] = 1
            else:
                inp = [i for i in inputs_character[index : index + sequence_length]]
                ouptwo = [inputs_character[index + sequence_length]]
                index += sequence_length + 1
                charin = ''.join([id2char[i] for i in inp])
                charou = ''.join([id2char[i] for i in ouptwo])
                inputs[np.arange(sequence_length), th] = inp
                labels[th, ouptwo[0]] = 1

            if len(inputs_character) - index <= 0:
                sentence = -9
        if e < start_iters:
            continue
        for j in range(sequence_length):
            for ind in range(num_layer):
                hidden_lstm_in[ind].append((hidden[ind], cell[ind]))
            qzg = np.array([quanzhong[int(id)] for id in inputs[j, :]])[:, np.newaxis]
            embedding_output = embedding.forward(inputs[j, :])
            flatten_col.append(embedding.flatten)
            hidden_cell_next, middle_in_col_0 = lstmlayers[0].forward(embedding_output, (hidden[0], cell[0]))
            hidden[0], cell[0] = hidden_cell_next
            hidden_cell_next, middle_in_col_1 = lstmlayers[1].forward(hidden[0], (hidden[1], cell[1]))
            hidden[1], cell[1] = hidden_cell_next

            hidden_col_in.append(hidden[1])
            middle_in_col[0].append(middle_in_col_0)
            middle_in_col[1].append(middle_in_col_1)
            if j==sequence_length-1:
                y = fullconnect.forward(hidden[1])
                loss, delta, predict = cross_entropy_loss(y, labels[:, :])
                delta = qzg * delta
            inputs_col[0].append(embedding_output)
            inputs_col[1].append(hidden[0])
            if j==sequence_length-1:
                loss_col.append(loss)
                delta_col.append(delta)
                predict_col.append(predict)

        for j in np.arange(sequence_length-1, -1, -1):
            if j==sequence_length-1:
                d_h_i1 = fullconnect.backward(delta_col[0], hidden_col_in[j])
            else:
                d_h_i1 = np.zeros_like(d_h_i1)
            _i_, _f_, _g_, _o_, c_tanhk = middle_in_col[1][j]
            input_delta1, hidden_delta[1], cell_delta[1] = lstmlayers[1].backward(d_h_i1, hidden_delta[1], cell_delta[1], inputs_col[1][j], hidden_lstm_in[1][j], _i_, _f_, _g_, _o_, c_tanhk)
            _i_, _f_, _g_, _o_, c_tanhk = middle_in_col[0][j]
            input_delta0, hidden_delta[0], cell_delta[0] = lstmlayers[0].backward(input_delta1, hidden_delta[0], cell_delta[0], inputs_col[0][j], hidden_lstm_in[0][j], _i_, _f_, _g_, _o_, c_tanhk)
            delta_embedding               = embedding.backward(input_delta0, flatten_col[j])

        if e % showiter==0:
            result = ''
            # k = inputs[0, -1, :]
            # kk = id2char[np.argmax(k)]
            hidden = []
            cell = []
            for ind in range(num_layer):
                hidden.append(np.zeros((1, hidden_size[ind])))
                cell.append(np.zeros((1, hidden_size[ind])))
            for j in range(sequence_length):
                first_inputs = inputs[j, -1]
                kk = id2char[first_inputs]
                embedding_output = embedding.forward(inputs[j, -1])
                (hidden[0], cell[0]), middle_output = lstmlayers[0].forward(embedding_output, (hidden[0], cell[0]))
                (hidden[1], cell[1]), middle_output = lstmlayers[1].forward(hidden[0], (hidden[1], cell[1]))
                if j==sequence_length-1:
                    y = fullconnect.forward(hidden[1])
                    p_shift = y - np.max(y, axis = -1)[:, np.newaxis]   # avoid too large in exp
                    predict = np.exp(p_shift) / np.sum(np.exp(p_shift), axis = -1)[:, np.newaxis]
                    p = np.argmax(predict, axis=-1)[0]
                    rek = id2char[p]
                    result += rek
                    if rek==end:
                        break
        embedding.update(lr)
        embedding.setzero()
        for ind in range(num_layer):
            lstmlayers[ind].update(lr)
            lstmlayers[ind].setzero()

        fullconnect.update(lr)
        fullconnect.setzero()
        meanloss = str(np.mean(loss_col))
        meanloss = str(lr)[:6] + " " + str(iters) + " " + meanloss[:10]

        if (e + 1) % savemodel == 0:
            savemodel_fun(embedding, num_layer, lstmlayers, fullconnect, e)
        if e % showiter==0:
            print(e, meanloss, charin, charou, result)
    savemodel_fun(embedding, num_layer, lstmlayers, fullconnect, e)
    return None

def savemodel_fun(embedding, num_layer, lstmlayers, fullconnect, e):
    model = []
    model.append(embedding.save_model())
    for ind in range(num_layer):
        model.append(lstmlayers[ind].save_model())
    model.append(fullconnect.save_model())
    model.append(e)
    with open(pretrained_model, 'wb') as obj:
        pickle.dump(model, obj)

if __name__=="__main__":
    frequency = 2000
    pretrained_model = os.path.join(abspath, "models", 'tangshi_lstm2layer_embedding_%d.pkl'%frequency)
    train()