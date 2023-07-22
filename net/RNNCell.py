import numpy as np
import torch
from torch import nn
from copy import deepcopy

def torch_compare_rnn(input_size, delta, hidden_size, bias, inputs, inputs_params, hidden_params, bias_ih_params, bias_hh_params):
    network = nn.RNNCell(input_size=input_size, hidden_size=hidden_size, bias=bias).requires_grad_(True)
    network.double()
    cnt = 0
    for i in network.parameters():
        if cnt==0:
            i.data = torch.from_numpy(inputs_params)
            i.retain_grad = True
        elif cnt==1:
            i.data = torch.from_numpy(hidden_params)
            i.retain_grad = True
        elif cnt==2:
            i.data = torch.from_numpy(bias_ih_params)
            i.retain_grad = True
        else:
            i.data = torch.from_numpy(bias_hh_params)
            i.retain_grad = True
        cnt += 1
    inputs = torch.tensor(inputs, requires_grad=True, dtype=torch.float64)
    output = network(inputs)
    delta = torch.tensor(delta)
    output.backward(delta)
    # sum = torch.sum(output) # make sure the gradient is 1
    # kk = sum.backward()
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
    return output, k, grad_inputs_params, grad_hidden_params, grad_bias_ih, grad_bias_hh

class rnncell_layer(object):
    def __init__(self, input_size, hidden_size, bias, inputs_params=[], hidden_params=[], bias_ih_params=[], bias_hh_params=[]):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        if bias and list(bias_ih_params)!=[]:
            self.bias_ih_params = bias_ih_params[np.newaxis, :]
        elif bias:
            ranges = np.sqrt(1 / hidden_size)
            self.bias_ih_params = np.random.uniform(-ranges, ranges, (hidden_size))[np.newaxis, :]

        if bias and list(bias_hh_params)!=[]:
            self.bias_hh_params = bias_hh_params[np.newaxis, :]
        elif bias:
            ranges = np.sqrt(1 / hidden_size)
            self.bias_hh_params = np.random.uniform(-ranges, ranges, (hidden_size))[np.newaxis, :]

        if list(inputs_params)!=[]:
            self.inputs_params = inputs_params
        else:
            ranges = np.sqrt(1 / (hidden_size))
            self.inputs_params = np.random.uniform(-ranges, ranges, (input_size, hidden_size))

        if list(hidden_params)!=[]:
            self.hidden_params = hidden_params
        else:
            ranges = np.sqrt(1 / (hidden_size))
            self.hidden_params = np.random.uniform(-ranges, ranges, (hidden_size, hidden_size))

        self.inputs_params_delta = np.zeros((input_size, hidden_size)).astype(np.float64)
        self.hidden_params_delta = np.zeros((hidden_size, hidden_size)).astype(np.float64)
        self.bias_ih_delta = np.zeros(hidden_size).astype(np.float64)
        self.bias_hh_delta = np.zeros(hidden_size).astype(np.float64)

    def forward(self, inputs, hidden0):
        self.z_t = np.matmul(inputs, self.inputs_params) + np.matmul(hidden0, self.hidden_params)
        self.h_t_1 = np.tanh(self.z_t + self.bias_ih_params + self.bias_hh_params)
        # self.h_t_1 = self.z_t + self.bias_ih_params + self.bias_hh_params
        return self.h_t_1

    def backward(self, delta, hidden_delta, inputs, hidden0, h_t_1):
        # previous layer delta
        delta += hidden_delta
        d_z_t       = delta * (1 - h_t_1**2)
        input_delta = np.matmul(d_z_t, self.inputs_params.T)
        self.inputs_params_delta += np.matmul(d_z_t.T, inputs).T
        hidden_delta = np.matmul(d_z_t, self.hidden_params.T)
        self.hidden_params_delta += np.matmul(d_z_t.T, hidden0).T

        self.bias_ih_delta += np.sum(d_z_t, axis=(0))
        self.bias_hh_delta += np.sum(d_z_t, axis=(0))

        return input_delta, hidden_delta

    def setzero(self):
        self.inputs_params_delta[...] = 0
        self.hidden_params_delta[...] = 0
        if self.bias:
            self.bias_ih_delta[...] = 0
            self.bias_hh_delta[...] = 0
        
    def update(self, lr=1e-10):
        self.inputs_params_delta = np.clip(self.inputs_params_delta, -6, 6)
        self.hidden_params_delta = np.clip(self.hidden_params_delta, -6, 6)
        if self.bias:
            self.bias_ih_delta = np.clip(self.bias_ih_delta, -6, 6)
            self.bias_hh_delta = np.clip(self.bias_hh_delta, -6, 6)
            self.bias_ih_params -= lr * self.bias_ih_delta[np.newaxis, :]
            self.bias_hh_params -= lr * self.bias_hh_delta[np.newaxis, :]
        self.inputs_params -= lr * self.inputs_params_delta
        self.hidden_params -= lr * self.hidden_params_delta

    def save_model(self):
        return [self.inputs_params.astype(np.float32),  \
                self.hidden_params.astype(np.float32),  \
                self.bias_ih_params.astype(np.float32), \
                self.bias_hh_params.astype(np.float32)]

    def restore_model(self, models):
        self.inputs_params = models[0]
        self.hidden_params = models[1]
        self.bias_ih_params = models[2]
        self.bias_hh_params = models[3]

def train_single():
    input_size  = 100
    hidden_size = 900
    batch_size = 100
    bias = True
    inputs = np.random.randn(batch_size, input_size) 
    inputs_params = np.random.rand(input_size, hidden_size)  / np.sqrt(input_size/2)
    hidden_params = np.random.rand(hidden_size, hidden_size)   / np.sqrt(hidden_size/2)
    if bias:
        bias_ih_params = np.random.rand(hidden_size)    / np.sqrt(hidden_size/2)
        bias_hh_params = np.random.rand(hidden_size)    / np.sqrt(hidden_size/2)
    else:
        bias_ih_params = []
        bias_hh_params = []

    hidden_0 = np.random.rand(batch_size, hidden_size)
    hidden_delta = np.zeros((batch_size, hidden_size))

    rnncell = rnncell_layer(input_size, hidden_size, bias, inputs_params, hidden_params, bias_ih_params, bias_hh_params)
    outputs = np.random.rand(batch_size, hidden_size)
    # output = rnncell.forward(inputs, hidden_0)
    # delta = np.ones((batch_size, hidden_size)).astype(np.float64)
    # partial, hidden_delta = rnncell.backward(delta.T, hidden_delta.T, inputs.T, hidden_0.T)
    for i in range(3000):
        out = rnncell.forward(inputs, hidden_0)
        sum = np.sum((outputs - out) * (outputs - out))
        delta = 2*(out - outputs)
        partial, _ = rnncell.backward(delta, hidden_delta, inputs, hidden_0, outputs)
        rnncell.update(0.0001)
        rnncell.setzero()
        # out = convolution.forward(inputs)
        # sum = np.sum((outputs - out) * (outputs - out))
        # delta = 2*(out - outputs)
        # partial_, = convolution.backward_common(delta)
        # partial = convolution.backward(delta, 0.0001)
        print(sum)

if __name__=="__main__":
    #https://blog.csdn.net/SHU15121856/article/details/104387209
    #https://gist.github.com/karpathy/d4dee566867f8291f086
    #https://github.com/JY-Yoon/RNN-Implementation-using-NumPy/blob/master/RNN%20Implementation%20using%20NumPy.ipynb
    #https://stackoverflow.com/questions/47868265/what-is-the-difference-between-an-embedding-layer-and-a-dense-layer
    #https://towardsdatascience.com/deep-learning-4-embedding-layers-f9a02d55ac12
    #https://discuss.pytorch.org/t/how-nn-embedding-trained/32533/5
    # https://github.com/krocki/dnc/blob/master/rnn-numpy
    # https://github.com/CaptainE/RNN-LSTM-in-numpy
    # https://discuss.pytorch.org/t/what-is-num-layers-in-rnn-module/9843

    # train_single()

    input_size  = 1000
    hidden_size = 200
    batch_size = 10
    bias = True
    inputs = np.random.randn(batch_size, input_size)
    inputs_params = np.random.rand(input_size, hidden_size)
    hidden_params = np.random.rand(hidden_size, hidden_size)
    if bias:
        bias_ih_params = np.random.rand(hidden_size)
        bias_hh_params = np.random.rand(hidden_size)
    else:
        bias_ih_params = []
        bias_hh_params = []

    hidden_0 = np.zeros((batch_size, hidden_size))
    hidden_delta = np.zeros((batch_size, hidden_size))

    rnncell = rnncell_layer(input_size, hidden_size, bias, inputs_params, hidden_params, bias_ih_params, bias_hh_params)
    output = rnncell.forward(inputs, hidden_0)
    delta = np.ones((batch_size, hidden_size)).astype(np.float64)
    partial, hidden_delta = rnncell.backward(delta, hidden_delta, inputs, hidden_0, output)
    
    output_torch, partial_torch, grad_inputs_params, grad_hidden_params, grad_bias_ih, grad_bias_hh = torch_compare_rnn(input_size, delta, hidden_size, bias, inputs, inputs_params.T, hidden_params.T, bias_ih_params, bias_hh_params)
    assert np.mean(np.abs(output - output_torch.cpu().detach().numpy())) < 1e-6, np.mean(np.abs(output - output_torch.cpu().detach().numpy()))
    assert np.mean(np.abs(partial - partial_torch.cpu().detach().numpy())) < 1e-6, np.mean(np.abs(partial - partial_torch.cpu().detach().numpy()))
    assert np.mean(np.abs(rnncell.inputs_params_delta.T - grad_inputs_params.cpu().detach().numpy())) < 1e-6, np.mean(np.abs(rnncell.inputs_params_delta.T - grad_inputs_params.cpu().detach().numpy()))
    assert np.mean(np.abs(rnncell.hidden_params_delta.T - grad_hidden_params.cpu().detach().numpy())) < 1e-6, np.mean(np.abs(rnncell.hidden_params_delta.T - grad_hidden_params.cpu().detach().numpy()))
    assert np.mean(np.abs(rnncell.bias_ih_delta - grad_bias_ih.cpu().detach().numpy())) < 1e-6, np.mean(np.abs(rnncell.bias_ih_delta - grad_bias_ih.cpu().detach().numpy()))
    assert np.mean(np.abs(rnncell.bias_hh_delta - grad_bias_hh.cpu().detach().numpy())) < 1e-6, np.mean(np.abs(rnncell.bias_hh_delta - grad_bias_hh.cpu().detach().numpy()))