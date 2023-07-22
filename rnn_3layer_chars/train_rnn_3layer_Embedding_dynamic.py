import os
import numpy as np
import sys
import pickle
import json
from copy import deepcopy
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
    jsonpth = os.path.join(abspath, 'dataset', r"tangshi_rnn3layer_dynamic.json")
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
    hidden_size = [600, 300, 160]
    batch_size = 1
    sequence_length = 6
    num_layer = 3
    bias = True

    epoch = 20
    cnt = 600000 - 160000 + 10000
    # for i in all_lines:
    #     if i=="，" or i=="。" or i=="？":
    #         cnt+=1
    iters = cnt * epoch // batch_size
    showiter = 100
    savemodel = 1000
    learning_rate = 0.001
    
    embedding = Embedding_layer(length, embedding_dim = embedding_dim)
    rnn_layer0 = rnncell_layer(embedding_dim, hidden_size[0], bias)
    rnn_layer1 = rnncell_layer(hidden_size[0], hidden_size[1], bias)
    rnn_layer2 = rnncell_layer(hidden_size[1], hidden_size[2], bias)
    rnnlayers = [rnn_layer0, rnn_layer1, rnn_layer2]
    
    rnn_decode0 = rnncell_layer(embedding_dim, hidden_size[0], bias)
    rnn_decode1 = rnncell_layer(hidden_size[0], hidden_size[1], bias)
    rnn_decode2 = rnncell_layer(hidden_size[1], hidden_size[2], bias)
    rnn_decode = [rnn_decode0, rnn_decode1, rnn_decode2]
    fullconnect = fclayer(hidden_size[2], length, True)
    
    start_iters  = 0
    if os.path.exists(pretrained_model):
        with open(pretrained_model, 'rb') as obj:
            models = pickle.load(obj)
        embedding.restore_model(models[0])
        rnnlayers[0].restore_model(models[1])
        rnnlayers[1].restore_model(models[2])
        rnnlayers[2].restore_model(models[3])
        rnn_decode[0].restore_model(models[2*2])
        rnn_decode[1].restore_model(models[2*2+1])
        rnn_decode[2].restore_model(models[6])
        fullconnect.restore_model(models[7])
        start_iters = models[-1]
    else:
        # exit(-1)
        pass

    all_lines = all_lines.split("\n")
    lines = 0
    for e in range(iters):
        # if e < 1000:
        #     lr = 0.0001
        # else:
        lr = learning_rate

        if e == int(iters * (16/20)):
            lr = learning_rate * 0.1
        elif e == int(iters * (18/20)):
            lr = learning_rate * 0.1 * 0.1

        content = all_lines[lines]
        lines = (lines + 1) % len(all_lines)
        if e < start_iters:
            continue
        c0 = content.split("，")
        all_content = []
        max_len = -6
        for i in c0:
            c1 = i.split("。")
            for j in c1:
                c2 = j.split("？")
                while "" in c2:
                    c2.remove("")
                if len(c2)!=0:
                    for c2_ in c2:
                        index = content.find(c2_)
                        try:
                            tails = content[index + len(c2_)]
                        except:
                            tails = "。"
                        all_content.append([char2id[c2__] for c2__ in c2_])
                        all_content[-1] += [char2id[tails]]
                    max_len = max(len(c2[0]), max_len)

        for i in range(len(all_content) - 1):
            inputs = np.array(all_content[i])
            if i==len(all_content) - 2:
                output = np.array(all_content[i+1] + [endid])
            else:
                output = np.array(all_content[i+1])

            hidden_col_rnnin = [[] for i in range(num_layer)]
            hidden_col_rnnout = [[] for i in range(num_layer)]
            hidden_col_in =[]
            hidden_col_out =[]
            loss_col = []
            delta_col = []
            predict_col = []
            inputs_col_out = [[] for i in range(num_layer)]
            inputs_colin = [[] for i in range(num_layer)]
            hidden = []
            hidden_out = []
            hidden_delta = []
            flatten_col = []
            for ind in range(num_layer):
                hidden.append(np.zeros((batch_size, hidden_size[ind])))
                hidden_delta.append(np.zeros((batch_size, hidden_size[ind])))
            for j in range(len(inputs)):
                for ind in range(num_layer):
                    hidden_col_rnnin[ind].append(hidden[ind])

                ink = np.array([inputs[j]])
                embedding_output = embedding.forward(ink)
                flatten_col.append(embedding.flatten)
                hidden[0] = rnnlayers[0].forward(embedding_output, hidden[0])
                hidden[1] = rnnlayers[1].forward(hidden[0], hidden[1])
                hidden[2] = rnnlayers[2].forward(hidden[1], hidden[2])
                hidden_col_in.append(hidden[2])
                inputs_colin[0].append(embedding_output)
                inputs_colin[1].append(hidden[0])
                inputs_colin[2].append(hidden[1])

            zerosin = np.zeros_like(embedding_output)
            hidden_out.extend([deepcopy(hidden[0]), deepcopy(hidden[1]), deepcopy(hidden[2])])
            for j in range(len(output)):
                for ind in range(num_layer):
                    hidden_col_rnnout[ind].append(hidden_out[ind])
                hidden_out[0] = rnn_decode[0].forward(zerosin, hidden_out[0])
                hidden_out[1] = rnn_decode[1].forward(hidden_out[0], hidden_out[1])
                hidden_out[2] = rnn_decode[2].forward(hidden_out[1], hidden_out[2])

                hidden_col_out.append(hidden_out[2])
                
                y = fullconnect.forward(hidden_out[2])
                label = np.zeros((batch_size, length), dtype = np.int32)
                label[0, output[j]] = 1
                loss, delta, predict = cross_entropy_loss(y, label)
                inputs_col_out[0].append(zerosin)
                inputs_col_out[1].append(hidden_out[0])
                inputs_col_out[2].append(hidden_out[1])
                loss_col.append(loss)
                delta_col.append(delta)
                predict_col.append(predict)
                
            for j in range(len(output) - 1, -1, -1):
                d_h_i1 = fullconnect.backward(delta_col[j], hidden_col_out[j])
                input_delta2, hidden_delta[2] = rnn_decode[2].backward(d_h_i1, hidden_delta[2], inputs_col_out[2][j], hidden_col_rnnout[2][j], hidden_col_out[j])
                input_delta1, hidden_delta[1] = rnn_decode[1].backward(input_delta2, hidden_delta[1], inputs_col_out[1][j], hidden_col_rnnout[1][j], inputs_col_out[2][j])
                input_delta0, hidden_delta[0] = rnn_decode[0].backward(input_delta1, hidden_delta[0], inputs_col_out[0][j], hidden_col_rnnout[0][j], inputs_col_out[1][j])

            zeros_delta = np.zeros_like(d_h_i1)
            for j in range(len(inputs)-1, -1, -1):
                input_delta2, hidden_delta[2] = rnnlayers[2].backward(zeros_delta, hidden_delta[2], inputs_colin[2][j], hidden_col_rnnin[2][j], hidden_col_in[j])
                input_delta1, hidden_delta[1] = rnnlayers[1].backward(input_delta2, hidden_delta[1], inputs_colin[1][j], hidden_col_rnnin[1][j], inputs_colin[2][j])
                input_delta0, hidden_delta[0] = rnnlayers[0].backward(input_delta1, hidden_delta[0], inputs_colin[0][j], hidden_col_rnnin[0][j], inputs_colin[1][j])
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
                for ij in range(len(inputs)):
                    # first_inputs = inputs[j]
                    # kk = id2char[first_inputs]
                    embedding_output = embedding.forward(inputs[ij])
                    hidden[0] = rnnlayers[0].forward(embedding_output, hidden[0])
                    hidden[1] = rnnlayers[1].forward(hidden[0], hidden[1])
                    hidden[2] = rnnlayers[2].forward(hidden[1], hidden[2])
                
                zerosin = np.zeros_like(embedding_output)
                for _ in range(len(output)):
                    hidden[0] = rnn_decode[0].forward(zerosin, hidden[0])
                    hidden[1] = rnn_decode[1].forward(hidden[0], hidden[1])
                    hidden[2] = rnn_decode[2].forward(hidden[1], hidden[2])
                    
                    y = fullconnect.forward(hidden[2])
                    p_shift = y - np.max(y, axis = -1)[:, np.newaxis]   # avoid too large in exp
                    predict = np.exp(p_shift) / np.sum(np.exp(p_shift), axis = -1)[:, np.newaxis]
                    p = np.argmax(predict, axis=-1)[0]
                    rek = id2char[p]
                    # if rek==end:
                    #     break
                    result += rek
            charin = ''.join([id2char[ij] for ij in inputs])
            charou = "".join([id2char[ij] for ij in output])
            if (e + 1) % savemodel == 0:
                model = []
                model.append(embedding.save_model())
                for ind in range(num_layer):
                    model.append(rnnlayers[ind].save_model())
                for ind in range(num_layer):
                    model.append(rnn_decode[ind].save_model())
                model.append(fullconnect.save_model())
                model.append(e)
                with open(os.path.join(abspath, "models", 'tangshi_rnn3layer_embedding_dynamic_char.pkl'), 'wb') as obj:
                    pickle.dump(model, obj)
            if e % showiter==0:
                print(e, meanloss, charin, charou, result)

        model = []
        model.append(embedding.save_model())
        for ind in range(num_layer):
            model.append(rnnlayers[ind].save_model())
        for ind in range(num_layer):
            model.append(rnn_decode[ind].save_model())
        model.append(fullconnect.save_model())
        model.append(e)
        with open(os.path.join(abspath, "models", 'tangshi_rnn3layer_embedding_dynamic_char.pkl'), 'wb') as obj:
            pickle.dump(model, obj)

if __name__=="__main__":
    pretrained_model = os.path.join(abspath, "models", 'tangshi_rnn3layer_embedding_dynamic_char.pkl')
    train()