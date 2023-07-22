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

from net.RNNCell import rnncell_layer
from net.fullconnect import fclayer
from net.loss import cross_entropy_loss
from net.Embedding import Embedding_layer

def preprocess():
    jsonpth = os.path.join(abspath, 'dataset', r"tangshi_rnn2layer.json")
    inpath = os.path.join(abspath, 'dataset', r"tangshi.txt")
    all_lines = ""
    with open(inpath, 'r', encoding='utf-8') as obj:
        for i in obj.readlines():
            if '?' in i:
                continue
            all_lines += i
    end = "&"
    unique = set(list(all_lines))
    unique.add(end)
    unique.remove("\n")
    length = len(unique)
    id2char = {i:char for i, char in enumerate(unique)}
    char2id = {char:i for i, char in enumerate(unique)}
    endid = char2id['&']
    if not os.path.exists(jsonpth):
        with open(jsonpth, 'w', encoding='utf-8') as obj:
            json.dump({"id2char":id2char, 'char2id':char2id}, obj, indent=2, separators=(",", ":"))
    else:
        with open(jsonpth, 'r', encoding='utf-8') as obj:
            jsonfile = json.load(obj)
        id2chark = jsonfile["id2char"]
        char2id = jsonfile["char2id"]
        length = len(id2chark)
        id2char = {}
        for key, value in id2chark.items():
            id2char[int(key)] = value
        endid = char2id['&']

    return id2char, char2id, length, all_lines, endid, end

def train():
    id2char, char2id, length, all_lines, endid, end = preprocess()
    embedding_dim  = 100
    hidden_size = [300, 300]
    batch_size = 1
    sequence_length = 6
    num_layer = len(hidden_size)
    bias = True

    epoch = 20
    iters = (len(all_lines.replace("\n", "")) // (sequence_length + 1) + 2*sequence_length) * epoch // batch_size
    showiter = 100
    savemodel = 1000
    learning_rate = 0.001
    
    embedding = Embedding_layer(length, embedding_dim = embedding_dim)
    rnn_layer0 = rnncell_layer(embedding_dim, hidden_size[0], bias)
    rnn_layer1 = rnncell_layer(hidden_size[0], hidden_size[1], bias)
    rnnlayers = [rnn_layer0, rnn_layer1]
    fullconnect = fclayer(hidden_size[1], length, True)
    
    start_iters  = 0
    if os.path.exists(pretrained_model):
        with open(pretrained_model, 'rb') as obj:
            models = pickle.load(obj)
        embedding.restore_model(models[0])
        rnnlayers[0].restore_model(models[1])
        rnnlayers[1].restore_model(models[2])
        fullconnect.restore_model(models[3])
        start_iters = models[-1]
    else:
        # exit(-1)
        pass

    sentence = -6
    all_lines = all_lines.split("\n")
    choose_lines = -1
    for e in range(iters):
        hidden_col_rnn = [[] for i in range(num_layer)]
        hidden_col_fc =[]
        loss_col = []
        delta_col = []
        predict_col = []
        inputs_col = [[] for i in range(num_layer)]
        hidden = []
        hidden_delta = []
        flatten_col = []
        for ind in range(num_layer):
            hidden.append(np.zeros((batch_size, hidden_size[ind])))
            hidden_delta.append(np.zeros((batch_size, hidden_size[ind])))
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
                inp = [char2id[i] for i in inputs_character[index:-1]]
                inptwo = [endid for i in range(sequence_length - len(inp))]
                try:
                    oup = [char2id[inputs_character[-1]]]
                except:
                    oup = [endid]
                index += len(inputs_character)
                charin = ''.join([id2char[i] for i in inp + inptwo])
                charou = ''.join([id2char[i] for i in oup])
                inputs[np.arange(len(inp)), th] = inp
                inputs[np.arange(len(inp), sequence_length), th] = inptwo
                labels[th, oup[0]] = 1
            else:
                inp = [char2id[i] for i in inputs_character[index : index + sequence_length]]
                ouptwo = [char2id[inputs_character[index + sequence_length]]]
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
                hidden_col_rnn[ind].append(hidden[ind])

            embedding_output = embedding.forward(inputs[j, :])
            flatten_col.append(embedding.flatten)
            hidden[0] = rnnlayers[0].forward(embedding_output, hidden[0])
            hidden[1] = rnnlayers[1].forward(hidden[0], hidden[1])

            hidden_col_fc.append(hidden[1])
            if j==sequence_length-1:
                y = fullconnect.forward(hidden[1])
                loss, delta, predict = cross_entropy_loss(y, labels[:, :])
            inputs_col[0].append(embedding_output)
            inputs_col[1].append(hidden[0])
            if j==sequence_length-1:
                loss_col.append(loss)
                delta_col.append(delta)
                predict_col.append(predict)

        for j in np.arange(sequence_length-1, -1, -1):
            if j==sequence_length-1:
                d_h_i1 = fullconnect.backward(delta_col[0], hidden_col_fc[j])
            else:
                d_h_i1 = np.zeros_like(d_h_i1)
            input_delta1, hidden_delta[1] = rnnlayers[1].backward(d_h_i1, hidden_delta[1], inputs_col[1][j], hidden_col_rnn[1][j], hidden_col_fc[j])
            input_delta0, hidden_delta[0] = rnnlayers[0].backward(input_delta1, hidden_delta[0], inputs_col[0][j], hidden_col_rnn[0][j], inputs_col[1][j])
            delta_embedding               = embedding.backward(input_delta0, flatten_col[j])
        embedding.update(lr)
        embedding.setzero()
        for ind in range(num_layer):
            rnnlayers[ind].update(lr)
            rnnlayers[ind].setzero()

        fullconnect.update(lr)
        fullconnect.setzero()
        meanloss = str(np.mean(loss_col))
        meanloss = str(lr)[:6] + " " + str(iters) + " " + meanloss[:10]

        if e % showiter==0:
            result = ''
            # k = inputs[0, -1, :]
            # kk = id2char[np.argmax(k)]
            hidden = []
            for ind in range(num_layer):
                hidden.append(np.zeros((1, hidden_size[ind])))
            for j in range(sequence_length):
                first_inputs = inputs[j, -1]
                kk = id2char[first_inputs]
                embedding_output = embedding.forward(inputs[j, -1])
                hidden[0] = rnnlayers[0].forward(embedding_output, hidden[0])
                hidden[1] = rnnlayers[1].forward(hidden[0], hidden[1])
                if j==sequence_length-1:
                    y = fullconnect.forward(hidden[1])
                    p_shift = y - np.max(y, axis = -1)[:, np.newaxis]   # avoid too large in exp
                    predict = np.exp(p_shift) / np.sum(np.exp(p_shift), axis = -1)[:, np.newaxis]
                    p = np.argmax(predict, axis=-1)[0]
                    rek = id2char[p]
                    result += rek
                    if rek==end:
                        break
        if (e + 1) % savemodel == 0:
            model = []
            model.append(embedding.save_model())
            for ind in range(num_layer):
                model.append(rnnlayers[ind].save_model())
            model.append(fullconnect.save_model())
            model.append(e)
            with open(os.path.join(abspath, "models", 'tangshi_rnn2layer_embedding_char.pkl'), 'wb') as obj:
                pickle.dump(model, obj)
        if e % showiter==0:
            print(e, meanloss, charin, charou, result)

    model = []
    model.append(embedding.save_model())
    for ind in range(num_layer):
        model.append(rnnlayers[ind].save_model())
    model.append(fullconnect.save_model())
    model.append(e)
    with open(os.path.join(abspath, "models", 'tangshi_rnn2layer_embedding_char.pkl'), 'wb') as obj:
        pickle.dump(model, obj)

if __name__=="__main__":
    pretrained_model = os.path.join(abspath, "models", 'tangshi_rnn2layer_embedding_char.pkl')
    train()