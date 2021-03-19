from tensorflow.keras.models import load_model
import os, cv2
import numpy as np
import torch
from torch_model import Actor
from time import time

def load_weights():
    model = load_model('simpleModel.h5')
    keras_weights = model.get_weights()

    net = Actor(1,1,time(),32)
    print(net)

    print('layers shape')
    params = net.state_dict()
    for idx, layer in enumerate(params.values()):
        print('keras Weights', keras_weights[idx].shape, 'and Pytorch Weights', layer.shape)
    #print(layer.data)
    return net, keras_weights

def convert_weight(net, keras_weights):
    print('load Keras Weight into PyTorch model')
    net.fc1.weight.date = torch.from_numpy(np.transpose(keras_weights[0]))
    net.fc1.bias.date = torch.from_numpy(keras_weights[1])
    net.fc2.weight.date = torch.from_numpy(np.transpose(keras_weights[2]))
    net.fc2.bias.date = torch.from_numpy(keras_weights[3])
    net.fc3.weight.date = torch.from_numpy(np.transpose(keras_weights[4]))
    net.fc3.bias.date = torch.from_numpy(keras_weights[5])
    net.fc4.weight.date = torch.from_numpy(np.transpose(keras_weights[6]))
    net.fc4.bias.date = torch.from_numpy(keras_weights[7])
    net.fc5.weight.date = torch.from_numpy(np.transpose(keras_weights[8]))
    net.fc5.bias.date = torch.from_numpy(keras_weights[9])
    return net

def main():
    inp = torch.Tensor(1, 32, 1)
    net, keras_weights = load_weights()
    out1 = net(inp)
    print(out1)
    net = convert_weight(net, keras_weights)
    out2 = net(inp)
    print(out2)

if __name__=='__main__':
    main()

