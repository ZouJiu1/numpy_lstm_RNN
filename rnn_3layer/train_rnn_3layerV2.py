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
from tokenlzh import preprocess

def train():
    id2char, char2id, length, all_lines, endid, end, quanzhong = preprocess()
    # inputs_character = r"abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz "
    # unique = set(list(inputs_character))
    # length = len(unique)
    # id2char = {i:char for i, char in enumerate(unique)}
    # char2id = {char:i for i, char in enumerate(unique)}

    # with open(os.path.join(abspath, r"dataset", r"tangshi_rnn3layer.json"), 'w') as obj:
    #     json.dump({"id2char":id2char, 'char2id':char2id}, obj, indent=2, separators=(",", ":"))

    input_size  = length
    hidden_size = [600, 300, 160]
    batch_size = 2*2
    sequence_length = 6
    num_layer = 3
    bias = True

    epoch = 20
    iters = (len(all_lines.replace("\n", "")) // (sequence_length + 1)) * epoch // batch_size
    showiter = 1
    savemodel = 1000
    learning_rate = 0.1
    rnn_layer0 = rnncell_layer(input_size, hidden_size[0], bias)
    rnn_layer1 = rnncell_layer(hidden_size[0], hidden_size[1], bias)
    rnn_layer2 = rnncell_layer(hidden_size[1], hidden_size[2], bias)
    rnnlayers = [rnn_layer0, rnn_layer1, rnn_layer2]
    fullconnect = fclayer(hidden_size[2], input_size, True)

    start_iters  = 0
    if os.path.exists(pretrained_model):
        with open(pretrained_model, 'rb') as obj:
            models = pickle.load(obj)
        rnnlayers[0].restore_model(models[1])
        rnnlayers[1].restore_model(models[2])
        rnnlayers[2].restore_model(models[3])
        fullconnect.restore_model(models[3])
        start_iters = models[-1]
    else:
        # exit(-1)
        pass

    sentence = -6
    all_lines = all_lines.split("\n")
    choose_lines = -1
    for e in range(start_iters, iters):
        hidden_col_rnn = [[] for i in range(num_layer)]
        hidden_col_fc =[]
        loss_col = []
        delta_col = []
        predict_col = []
        inputs_col = [[] for i in range(num_layer)]
        hidden = []
        hidden_delta = []
        for ind in range(num_layer):
            hidden.append(np.zeros((batch_size, hidden_size[ind])))
            hidden_delta.append(np.zeros((batch_size, hidden_size[ind])))
            
        lr = learning_rate
        if e == int(iters * (16/20)):
            lr = learning_rate * 0.1
        elif e == int(iters * (18/20)):
            lr = learning_rate * 0.1 * 0.1

        inputs = np.zeros((sequence_length, batch_size, input_size))
        labels = np.zeros((sequence_length, batch_size, input_size))
        for th in range(batch_size):
            if sentence < 0:
                # choose_lines = np.random.randint(0, len(all_lines))
                choose_lines = (choose_lines + 1) % len(all_lines)
                inputs_character = all_lines[choose_lines]
                sentence = 9
                index = 0

            if len(inputs_character) - index < sequence_length + 1:
                sentence = -9
                inp = [char2id[i] for i in inputs_character[index:-1]]
                inptwo = [endid for i in range(sequence_length - len(inp))]
                oup = [char2id[i] for i in inputs_character[index+1:]]
                ouptwo = [endid for i in range(sequence_length - len(oup))]
                index += sequence_length + 1
                charin = ''.join([id2char[i] for i in inp])
                charou = ''.join([id2char[i] for i in oup])
                inputs[np.arange(len(inp)), th, inp] = 1
                inputs[np.arange(len(inp), sequence_length), th, inptwo] = 1
                labels[np.arange(len(oup)), th, oup] = 1
                labels[np.arange(len(oup), sequence_length), th, ouptwo] = 1
            else:
                inp = [char2id[i] for i in inputs_character[index : index + sequence_length]]
                oup = [char2id[i] for i in inputs_character[index + 1: index + sequence_length + 1]]
                index += sequence_length + 1
                charin = ''.join([id2char[i] for i in inp])
                charou = ''.join([id2char[i] for i in oup])
                inputs[np.arange(sequence_length), th, inp] = 1
                labels[np.arange(sequence_length), th, oup] = 1

            if len(inputs_character) - index <= 0:
                sentence = -9

        for j in range(sequence_length):
            for ind in range(num_layer):
                hidden_col_rnn[ind].append(hidden[ind])

            hidden[0] = rnnlayers[0].forward(inputs[j, :, :], hidden[0])
            hidden[1] = rnnlayers[1].forward(hidden[0], hidden[1])
            hidden[2] = rnnlayers[2].forward(hidden[1], hidden[2])

            hidden_col_fc.append(hidden[2])
            y = fullconnect.forward(hidden[2])

            loss, delta, predict = cross_entropy_loss(y, labels[j, :, :])
            inputs_col[0].append(inputs[j, :, :])
            inputs_col[1].append(hidden[0])
            inputs_col[2].append(hidden[1])
            loss_col.append(loss)
            delta_col.append(delta)
            predict_col.append(predict)

        for j in np.arange(sequence_length-1, -1, -1):
            d_h_i1 = fullconnect.backward(delta_col[j], hidden_col_fc[j])
            input_delta2, hidden_delta[2] = rnnlayers[2].backward(d_h_i1, hidden_delta[2], inputs_col[2][j], hidden_col_rnn[2][j], hidden_col_fc[j])
            input_delta1, hidden_delta[1] = rnnlayers[1].backward(input_delta2, hidden_delta[1], inputs_col[1][j], hidden_col_rnn[1][j], inputs_col[2][j])
            input_delta0, hidden_delta[0] = rnnlayers[0].backward(input_delta1, hidden_delta[0], inputs_col[0][j], hidden_col_rnn[0][j], inputs_col[1][j])

        for ind in range(num_layer):
            rnnlayers[ind].update(lr)
            rnnlayers[ind].setzero()

        fullconnect.update(lr)
        fullconnect.setzero()
        meanloss = str(np.mean(loss_col))
        meanloss = str(lr)[:6] + " " + str(iters) + " " + meanloss[:10]

        if e%showiter==0:
            result = ''
            # first_inputs = inputs[0, -1, :][np.newaxis, :]
            hidden = []
            for ind in range(num_layer):
                hidden.append(np.zeros((1, hidden_size[ind])))
            for j in range(sequence_length):
                first_inputs = inputs[j, -1, :][np.newaxis, :]
                # kk = id2char[np.argmax(inputs[j, -1, :])]
                hidden[0] = rnnlayers[0].forward(first_inputs, hidden[0])
                hidden[1] = rnnlayers[1].forward(hidden[0], hidden[1])
                hidden[2] = rnnlayers[2].forward(hidden[1], hidden[2])
                y = fullconnect.forward(hidden[2])
                p_shift = y - np.max(y, axis = -1)[:, np.newaxis]   # avoid too large in exp
                predict = np.exp(p_shift) / np.sum(np.exp(p_shift), axis = -1)[:, np.newaxis]
                p = np.argmax(predict, axis=-1)[0]
                # first_inputs  = np.zeros((1, input_size))
                # first_inputs[0, p] = 1
                rek = id2char[p]
                if rek==end:
                    break
                result += rek

        if (e + 1) % savemodel == 0:
            model = []
            for ind in range(num_layer):
                model.append(rnnlayers[ind].save_model())
            model.append(e)
            model.append(fullconnect.save_model())
            with open(os.path.join(abspath, "models", 'tangshi_rnn3layerV2.pkl'), 'wb') as obj:
                pickle.dump(model, obj)

        if e%showiter==0:
            print(e, meanloss, charin, charou, result)

    model = []
    for ind in range(num_layer):
        model.append(rnnlayers[ind].save_model())
    model.append(fullconnect.save_model())
    model.append(e)
    with open(os.path.join(abspath, "models", 'tangshi_rnn3layerV2.pkl'), 'wb') as obj:
        pickle.dump(model, obj)

if __name__=="__main__":
    pretrained_model = os.path.join(abspath, "models", 'tangshi_rnn3layerV2.pkl')
    train()