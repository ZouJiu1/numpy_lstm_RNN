import numpy as np
import torch
from torch import nn
from copy import deepcopy

def torch_compare_lstm(input_size, delta, hidden_size, bias, inputs, hidden0_cell0, \
    inputs_i_params, hidden_i_params, bias_ii_params, bias_hi_params, \
    inputs_f_params, hidden_f_params, bias_if_params, bias_hf_params, \
    inputs_g_params, hidden_g_params, bias_ig_params, bias_hg_params, \
    inputs_o_params, hidden_o_params, bias_io_params, bias_ho_params, \
    ):

    network = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size, bias=bias).requires_grad_(True)
    network.double()
    cnt = 0
    inputs_ih_params  = np.concatenate([inputs_i_params, inputs_f_params, inputs_g_params, inputs_o_params], axis=0)
    hidden_hh_params  = np.concatenate([hidden_i_params, hidden_f_params, hidden_g_params, hidden_o_params], axis=0)
    bias_ih_params = np.concatenate([bias_ii_params, bias_if_params, bias_ig_params, bias_io_params], axis=0)
    bias_hh_params  = np.concatenate([bias_hi_params, bias_hf_params, bias_hg_params, bias_ho_params], axis=0)
    for i in network.parameters():
        if cnt==0:
            i.data = torch.from_numpy(inputs_ih_params)
            i.retain_grad = True
        elif cnt==1:
            i.data = torch.from_numpy(hidden_hh_params)
            i.retain_grad = True
        elif cnt==2:
            i.data = torch.from_numpy(bias_ih_params)
            i.retain_grad = True
        else:
            i.data = torch.from_numpy(bias_hh_params)
            i.retain_grad = True
        cnt += 1
    inputs = torch.tensor(inputs, requires_grad=True, dtype=torch.float64)
    hidden0 = torch.tensor(hidden0_cell0[0], requires_grad=True, dtype=torch.float64)
    cell0 = torch.tensor(hidden0_cell0[1], requires_grad=True, dtype=torch.float64)
    output, cell_next = network(inputs, (hidden0, cell0))
    delta = torch.tensor(delta)
    output.backward(delta)
    # cell_next.backward(delta)

    grad_inputs_params = 0
    grad_hidden_params = 0
    grad_bias_ih = 0
    grad_bias_hh = 0
    cnt = 0
    for i in network.parameters():
        if cnt==0:
            grad_inputs_params = i.grad
        elif cnt==1:
            grad_hidden_params = i.grad
        elif cnt==2:
            grad_bias_ih = i.grad
        else:
            grad_bias_hh = i.grad
        cnt += 1
    inputs.retain_grad()
    output.retain_grad()
    k = inputs.grad
    return output, cell_next, k, grad_inputs_params, grad_hidden_params, grad_bias_ih, grad_bias_hh

class lstmcell_layer(object):
    def __init__(self, input_size, hidden_size, bias, \
                 inputs_i_params=[], hidden_i_params=[], bias_ii_params=[], bias_hi_params=[], \
                 inputs_f_params=[], hidden_f_params=[], bias_if_params=[], bias_hf_params=[], \
                 inputs_g_params=[], hidden_g_params=[], bias_ig_params=[], bias_hg_params=[], \
                 inputs_o_params=[], hidden_o_params=[], bias_io_params=[], bias_ho_params=[]):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        if bias:
            if list(bias_ii_params)!=[]:
                self.bias_ii_params = bias_ii_params[np.newaxis, :]
            else:
                ranges = np.sqrt(1 / hidden_size)
                self.bias_ii_params = np.random.uniform(-ranges, ranges, (hidden_size))[np.newaxis, :]
            if list(bias_if_params)!=[]:
                self.bias_if_params = bias_if_params[np.newaxis, :]
            else:
                ranges = np.sqrt(1 / hidden_size)
                self.bias_if_params = np.random.uniform(-ranges, ranges, (hidden_size))[np.newaxis, :]
            if list(bias_ig_params)!=[]:
                self.bias_ig_params = bias_ig_params[np.newaxis, :]
            else:
                ranges = np.sqrt(1 / hidden_size)
                self.bias_ig_params = np.random.uniform(-ranges, ranges, (hidden_size))[np.newaxis, :]
            if list(bias_io_params)!=[]:
                self.bias_io_params = bias_io_params[np.newaxis, :]
            else:
                ranges = np.sqrt(1 / hidden_size)
                self.bias_io_params = np.random.uniform(-ranges, ranges, (hidden_size))[np.newaxis, :]

            if list(bias_hi_params)!=[]:
                self.bias_hi_params = bias_hi_params[np.newaxis, :]
            else:
                ranges = np.sqrt(1 / hidden_size)
                self.bias_hi_params = np.random.uniform(-ranges, ranges, (hidden_size))[np.newaxis, :]
            if list(bias_hf_params)!=[]:
                self.bias_hf_params = bias_hf_params[np.newaxis, :]
            else:
                ranges = np.sqrt(1 / hidden_size)
                self.bias_hf_params = np.random.uniform(-ranges, ranges, (hidden_size))[np.newaxis, :]
            if list(bias_hg_params)!=[]:
                self.bias_hg_params = bias_hg_params[np.newaxis, :]
            else:
                ranges = np.sqrt(1 / hidden_size)
                self.bias_hg_params = np.random.uniform(-ranges, ranges, (hidden_size))[np.newaxis, :]
            if list(bias_ho_params)!=[]:
                self.bias_ho_params = bias_ho_params[np.newaxis, :]
            else:
                ranges = np.sqrt(1 / hidden_size)
                self.bias_ho_params = np.random.uniform(-ranges, ranges, (hidden_size))[np.newaxis, :]

        if list(inputs_i_params)!=[]:
            self.inputs_i_params = inputs_i_params
        else:
            ranges = np.sqrt(1 / (hidden_size))
            self.inputs_i_params = np.random.uniform(-ranges, ranges, (input_size, hidden_size))
        if list(inputs_f_params)!=[]:
            self.inputs_f_params = inputs_f_params
        else:
            ranges = np.sqrt(1 / (hidden_size))
            self.inputs_f_params = np.random.uniform(-ranges, ranges, (input_size, hidden_size))
        if list(inputs_g_params)!=[]:
            self.inputs_g_params = inputs_g_params
        else:
            ranges = np.sqrt(1 / (hidden_size))
            self.inputs_g_params = np.random.uniform(-ranges, ranges, (input_size, hidden_size))
        if list(inputs_o_params)!=[]:
            self.inputs_o_params = inputs_o_params
        else:
            ranges = np.sqrt(1 / (hidden_size))
            self.inputs_o_params = np.random.uniform(-ranges, ranges, (input_size, hidden_size))

        if list(hidden_i_params)!=[]:
            self.hidden_i_params = hidden_i_params
        else:
            ranges = np.sqrt(1 / (hidden_size))
            self.hidden_i_params = np.random.uniform(-ranges, ranges, (hidden_size, hidden_size))
        if list(hidden_f_params)!=[]:
            self.hidden_f_params = hidden_f_params
        else:
            ranges = np.sqrt(1 / (hidden_size))
            self.hidden_f_params = np.random.uniform(-ranges, ranges, (hidden_size, hidden_size))
        if list(hidden_g_params)!=[]:
            self.hidden_g_params = hidden_g_params
        else:
            ranges = np.sqrt(1 / (hidden_size))
            self.hidden_g_params = np.random.uniform(-ranges, ranges, (hidden_size, hidden_size))
        if list(hidden_o_params)!=[]:
            self.hidden_o_params = hidden_o_params
        else:
            ranges = np.sqrt(1 / (hidden_size))
            self.hidden_o_params = np.random.uniform(-ranges, ranges, (hidden_size, hidden_size))

        self.inputs_i_params_delta = np.zeros((input_size, hidden_size)).astype(np.float64)
        self.inputs_f_params_delta = np.zeros((input_size, hidden_size)).astype(np.float64)
        self.inputs_g_params_delta = np.zeros((input_size, hidden_size)).astype(np.float64)
        self.inputs_o_params_delta = np.zeros((input_size, hidden_size)).astype(np.float64)

        self.hidden_i_params_delta = np.zeros((hidden_size, hidden_size)).astype(np.float64)
        self.hidden_f_params_delta = np.zeros((hidden_size, hidden_size)).astype(np.float64)
        self.hidden_g_params_delta = np.zeros((hidden_size, hidden_size)).astype(np.float64)
        self.hidden_o_params_delta = np.zeros((hidden_size, hidden_size)).astype(np.float64)

        self.bias_ii_delta = np.zeros(hidden_size).astype(np.float64)
        self.bias_if_delta = np.zeros(hidden_size).astype(np.float64)
        self.bias_ig_delta = np.zeros(hidden_size).astype(np.float64)
        self.bias_io_delta = np.zeros(hidden_size).astype(np.float64)

        self.bias_hi_delta = np.zeros(hidden_size).astype(np.float64)
        self.bias_hf_delta = np.zeros(hidden_size).astype(np.float64)
        self.bias_hg_delta = np.zeros(hidden_size).astype(np.float64)
        self.bias_ho_delta = np.zeros(hidden_size).astype(np.float64)
    
    def sigmoid(self, inputs):
        return 1 / (1 + np.exp(-inputs))

    def forward(self, inputs, hidden0_cell0):
        hidden0 = hidden0_cell0[0]
        cell_0 = hidden0_cell0[1]
        self.i_ = np.matmul(inputs, self.inputs_i_params) + np.matmul(hidden0, self.hidden_i_params) + self.bias_ii_params + self.bias_hi_params
        self.f_ = np.matmul(inputs, self.inputs_f_params) + np.matmul(hidden0, self.hidden_f_params) + self.bias_if_params + self.bias_hf_params
        self.g_ = np.matmul(inputs, self.inputs_g_params) + np.matmul(hidden0, self.hidden_g_params) + self.bias_ig_params + self.bias_hg_params
        self.o_ = np.matmul(inputs, self.inputs_o_params) + np.matmul(hidden0, self.hidden_o_params) + self.bias_io_params + self.bias_ho_params

        self.i = self.sigmoid(self.i_)
        self.f = self.sigmoid(self.f_)
        self.g = np.tanh(self.g_)
        self.o = self.sigmoid(self.o_)

        self.c = self.f * cell_0 + self.i * self.g
        self.c_tanhk  = np.tanh(self.c)
        self.h = self.o * self.c_tanhk
        return (self.h, self.c), (self.i, self.f, self.g, self.o, self.c_tanhk)

    def backward(self, delta, hidden_delta, cell_delta, inputs, hidden0_cell0, i, f, g, o, c_tanhk):
        hidden0      = hidden0_cell0[0]
        cell_0       = hidden0_cell0[1]
        delta       += hidden_delta
        next_c_delta = delta * o * (1 - c_tanhk**2) + cell_delta
        cell_delta   = next_c_delta * f

        delta_o_   = delta * c_tanhk * (1 - o) * o
        delta_f_   = next_c_delta * cell_0 * (1 - f) * f
        delta_i_   = next_c_delta * g * (1 - i) * i
        delta_g_   = next_c_delta * i * (1 - g**2)

        input_delta = np.matmul(delta_i_, self.inputs_i_params.T)
        input_delta += np.matmul(delta_f_, self.inputs_f_params.T)
        input_delta += np.matmul(delta_g_, self.inputs_g_params.T)
        input_delta += np.matmul(delta_o_, self.inputs_o_params.T)

        self.inputs_i_params_delta += np.matmul(delta_i_.T, inputs).T
        self.inputs_f_params_delta += np.matmul(delta_f_.T, inputs).T
        self.inputs_g_params_delta += np.matmul(delta_g_.T, inputs).T
        self.inputs_o_params_delta += np.matmul(delta_o_.T, inputs).T

        hidden_delta = np.matmul(delta_i_, self.hidden_i_params.T)
        hidden_delta += np.matmul(delta_f_, self.hidden_f_params.T)
        hidden_delta += np.matmul(delta_g_, self.hidden_g_params.T)
        hidden_delta += np.matmul(delta_o_, self.hidden_o_params.T)

        self.hidden_i_params_delta += np.matmul(delta_i_.T, hidden0).T
        self.hidden_f_params_delta += np.matmul(delta_f_.T, hidden0).T
        self.hidden_g_params_delta += np.matmul(delta_g_.T, hidden0).T
        self.hidden_o_params_delta += np.matmul(delta_o_.T, hidden0).T

        self.bias_ii_delta += np.sum(delta_i_, axis=(0))
        self.bias_if_delta += np.sum(delta_f_, axis=(0))
        self.bias_ig_delta += np.sum(delta_g_, axis=(0))
        self.bias_io_delta += np.sum(delta_o_, axis=(0))

        self.bias_hi_delta += np.sum(delta_i_, axis=(0))
        self.bias_hf_delta += np.sum(delta_f_, axis=(0))
        self.bias_hg_delta += np.sum(delta_g_, axis=(0))
        self.bias_ho_delta += np.sum(delta_o_, axis=(0))
        
        return input_delta, hidden_delta, cell_delta

    def setzero(self):
        self.inputs_i_params_delta[...] = 0
        self.inputs_f_params_delta[...] = 0
        self.inputs_g_params_delta[...] = 0
        self.inputs_o_params_delta[...] = 0

        self.hidden_i_params_delta[...] = 0
        self.hidden_f_params_delta[...] = 0
        self.hidden_g_params_delta[...] = 0
        self.hidden_o_params_delta[...] = 0

        if self.bias:
            self.bias_ii_delta[...] = 0
            self.bias_if_delta[...] = 0
            self.bias_ig_delta[...] = 0
            self.bias_io_delta[...] = 0
            
            self.bias_hi_delta[...] = 0
            self.bias_hf_delta[...] = 0
            self.bias_hg_delta[...] = 0
            self.bias_ho_delta[...] = 0
        
    def update(self, lr=1e-10):
        # self.inputs_params_delta = np.clip(self.inputs_params_delta, -6, 6)
        # self.hidden_params_delta = np.clip(self.hidden_params_delta, -6, 6)
        if self.bias:
            # self.bias_ih_delta = np.clip(self.bias_ih_delta, -6, 6)
            # self.bias_hh_delta = np.clip(self.bias_hh_delta, -6, 6)
            self.bias_ii_params -= lr * self.bias_ii_delta[np.newaxis, :]
            self.bias_if_params -= lr * self.bias_if_delta[np.newaxis, :]
            self.bias_ig_params -= lr * self.bias_ig_delta[np.newaxis, :]
            self.bias_io_params -= lr * self.bias_io_delta[np.newaxis, :]

            self.bias_hi_params -= lr * self.bias_hi_delta[np.newaxis, :]
            self.bias_hf_params -= lr * self.bias_hf_delta[np.newaxis, :]
            self.bias_hg_params -= lr * self.bias_hg_delta[np.newaxis, :]
            self.bias_ho_params -= lr * self.bias_ho_delta[np.newaxis, :]

        self.inputs_i_params -= lr * self.inputs_i_params_delta
        self.inputs_f_params -= lr * self.inputs_f_params_delta
        self.inputs_g_params -= lr * self.inputs_g_params_delta
        self.inputs_o_params -= lr * self.inputs_o_params_delta

        self.hidden_i_params -= lr * self.hidden_i_params_delta
        self.hidden_f_params -= lr * self.hidden_f_params_delta
        self.hidden_g_params -= lr * self.hidden_g_params_delta
        self.hidden_o_params -= lr * self.hidden_o_params_delta
        
    def save_model(self):
        return [ \
                self.inputs_i_params, self.inputs_f_params, self.inputs_g_params, self.inputs_o_params ,\
                self.hidden_i_params, self.hidden_f_params, self.hidden_g_params, self.hidden_o_params, \
                self.bias_ii_params, self.bias_if_params, self.bias_ig_params, self.bias_io_params, \
                self.bias_hi_params, self.bias_hf_params, self.bias_hg_params, self.bias_ho_params, \
              ]

    def restore_model(self, models):
        self.inputs_i_params = models[0]
        self.inputs_f_params = models[1]
        self.inputs_g_params = models[2]
        self.inputs_o_params = models[3]
        
        self.hidden_i_params = models[4]
        self.hidden_f_params = models[5]
        self.hidden_g_params = models[6]
        self.hidden_o_params = models[7]

        self.bias_ii_params = models[8]
        self.bias_if_params = models[9]
        self.bias_ig_params = models[10]
        self.bias_io_params = models[11]

        self.bias_hi_params = models[12]
        self.bias_hf_params = models[13]
        self.bias_hg_params = models[14]
        self.bias_ho_params = models[15]

def train_single():
    outputs = np.random.rand(batch_size, hidden_size)
    hidden_delta = np.zeros((batch_size, hidden_size))
    cell_delta = np.zeros((batch_size, hidden_size))
    hidden_0 = np.random.rand(batch_size, hidden_size)
    cell_0 = np.random.rand(batch_size, hidden_size)
    for _ in range(3000):
        (hidden, cell), middle_output = lstmcell.forward(inputs, (hidden_0, cell_0))
        i, f, g, o, c_tanhk = middle_output
        sum = np.sum((outputs - hidden) * (outputs - hidden))
        delta = 2 * (hidden - outputs)
        _, _, _ = lstmcell.backward(delta, hidden_delta, cell_delta, inputs, (hidden_0, cell_0), i, f, g, o, c_tanhk)
        lstmcell.update(0.001)
        lstmcell.setzero()
        print(sum)

if __name__=="__main__":
    #http://colah.github.io/posts/2015-08-Understanding-LSTMs/
    #https://blog.csdn.net/zhaojc1995/article/details/80572098
    
    input_size  = 1000
    hidden_size = 200
    batch_size = 10
    bias = True
    ranges = np.sqrt(1 / hidden_size)
    inputs_i_params = np.random.uniform(-ranges, ranges, (input_size, hidden_size))
    inputs_f_params = np.random.uniform(-ranges, ranges, (input_size, hidden_size))
    inputs_g_params = np.random.uniform(-ranges, ranges, (input_size, hidden_size))
    inputs_o_params = np.random.uniform(-ranges, ranges, (input_size, hidden_size))

    hidden_i_params = np.random.uniform(-ranges, ranges, (hidden_size, hidden_size))
    hidden_f_params = np.random.uniform(-ranges, ranges, (hidden_size, hidden_size))
    hidden_g_params = np.random.uniform(-ranges, ranges, (hidden_size, hidden_size))
    hidden_o_params = np.random.uniform(-ranges, ranges, (hidden_size, hidden_size))
    
    inputs = np.random.randn(batch_size, input_size)
    if bias:
        ranges = np.sqrt(1 / hidden_size)
        bias_ii_params = np.random.uniform(-ranges, ranges, (hidden_size))
        bias_if_params = np.random.uniform(-ranges, ranges, (hidden_size))
        bias_ig_params = np.random.uniform(-ranges, ranges, (hidden_size))
        bias_io_params = np.random.uniform(-ranges, ranges, (hidden_size))

        bias_hi_params = np.random.uniform(-ranges, ranges, (hidden_size))
        bias_hf_params = np.random.uniform(-ranges, ranges, (hidden_size))
        bias_hg_params = np.random.uniform(-ranges, ranges, (hidden_size))
        bias_ho_params = np.random.uniform(-ranges, ranges, (hidden_size))
    else:
        bias_ii_params = []
        bias_if_params = []
        bias_ig_params = []
        bias_io_params = []

        bias_hi_params = []
        bias_hf_params = []
        bias_hg_params = []
        bias_ho_params = []

    hidden_0 = np.random.randn(batch_size, hidden_size)
    cell_0 = np.random.randn(batch_size, hidden_size)
    hidden0_cell0 = (hidden_0, cell_0)

    hidden_delta = np.zeros((batch_size, hidden_size))
    cell_delta = np.zeros((batch_size, hidden_size))

    lstmcell = lstmcell_layer(input_size, hidden_size, bias, \
                          inputs_i_params, hidden_i_params, bias_ii_params, bias_hi_params, \
                          inputs_f_params, hidden_f_params, bias_if_params, bias_hf_params, \
                          inputs_g_params, hidden_g_params, bias_ig_params, bias_hg_params, \
                          inputs_o_params, hidden_o_params, bias_io_params, bias_ho_params, \
                         )

##########################################################################
    # train_single()
##########################################################################

    hidden_cell_next, middle_output = lstmcell.forward(inputs, hidden0_cell0)
    h, c = hidden_cell_next
    i, f, g, o, c_tanhk = middle_output
    delta = np.ones((batch_size, hidden_size)).astype(np.float64)
    input_delta, hidden_delta, cell_delta = lstmcell.backward(delta, hidden_delta, cell_delta, inputs, hidden0_cell0, i, f, g, o, c_tanhk)

    inputs_params_delta = np.concatenate([lstmcell.inputs_i_params_delta, lstmcell.inputs_f_params_delta, lstmcell.inputs_g_params_delta, lstmcell.inputs_o_params_delta], axis = 1)
    hidden_params_delta = np.concatenate([lstmcell.hidden_i_params_delta, lstmcell.hidden_f_params_delta, lstmcell.hidden_g_params_delta, lstmcell.hidden_o_params_delta], axis = 1)
    bias_ih_delta = np.concatenate([lstmcell.bias_ii_delta, lstmcell.bias_if_delta, lstmcell.bias_ig_delta, lstmcell.bias_io_delta], axis = 0)
    bias_hh_delta = np.concatenate([lstmcell.bias_hi_delta, lstmcell.bias_hf_delta, lstmcell.bias_hg_delta, lstmcell.bias_ho_delta], axis = 0)

    hidden_out, cell_out, partial_torch, grad_inputs_params, grad_hidden_params, grad_bias_ih, \
        grad_bias_hh = torch_compare_lstm(\
            input_size, delta, hidden_size, bias, inputs, hidden0_cell0, \
            inputs_i_params.T, hidden_i_params.T, bias_ii_params.T, bias_hi_params.T, \
            inputs_f_params.T, hidden_f_params.T, bias_if_params.T, bias_hf_params.T, \
            inputs_g_params.T, hidden_g_params.T, bias_ig_params.T, bias_hg_params.T, \
            inputs_o_params.T, hidden_o_params.T, bias_io_params.T, bias_ho_params.T, \
            )
    assert np.mean(np.abs(h - hidden_out.cpu().detach().numpy())) < 1e-6, np.mean(np.abs(h - hidden_out.cpu().detach().numpy()))
    assert np.mean(np.abs(c - cell_out.cpu().detach().numpy())) < 1e-6, np.mean(np.abs(c - cell_out.cpu().detach().numpy()))
    assert np.mean(np.abs(input_delta - partial_torch.cpu().detach().numpy())) < 1e-6, np.mean(np.abs(input_delta - partial_torch.cpu().detach().numpy()))
    assert np.mean(np.abs(inputs_params_delta.T - grad_inputs_params.cpu().detach().numpy())) < 1e-6, np.mean(np.abs(inputs_params_delta.T - grad_inputs_params.cpu().detach().numpy()))
    assert np.mean(np.abs(hidden_params_delta.T - grad_hidden_params.cpu().detach().numpy())) < 1e-6, np.mean(np.abs(hidden_params_delta.T - grad_hidden_params.cpu().detach().numpy()))
    assert np.mean(np.abs(bias_ih_delta - grad_bias_ih.cpu().detach().numpy())) < 1e-6, np.mean(np.abs(bias_ih_delta - grad_bias_ih.cpu().detach().numpy()))
    assert np.mean(np.abs(bias_hh_delta - grad_bias_hh.cpu().detach().numpy())) < 1e-6, np.mean(np.abs(bias_hh_delta - grad_bias_hh.cpu().detach().numpy()))