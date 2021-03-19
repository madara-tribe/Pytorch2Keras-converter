import cv2
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
from converter import pytorch_to_keras
from torch_models.resnet18 import FullyConvolutionalResnet18
from tensorflow.keras.models import Model

def torch2keras_convert(model, path):
    image = cv2.imread(path)
    image = np.transpose(np.expand_dims(image, 0), [0, 3, 1, 2])
    torch_image = torch.Tensor(image)
    print(torch_image.shape)
    preds = model(torch_image)
    preds = torch.softmax(preds, dim=1)
    # Find the class with the maximum score in the n x m output map
    pred, class_idx = torch.max(preds, dim=1)
    #print("torch model pred index", class_idx)
    summary(model, input_size=torch_image.shape[-3:])
    
    print('converting Pythorch model to keras model')
    keras_model = pytorch_to_keras(model, torch_image, [torch_image.shape[-3:]], verbose=False)
    keras_model.summary()
    return keras_model
    
if __name__=='__main__':
    GPU_ID = '0'
    device = torch.device("cuda:{}".format(GPU_ID) if torch.cuda.is_available() else "cpu")
    path = 'images/sample.jpg'
    model = FullyConvolutionalResnet18(pretrained=True).eval()
    keras_model = torch2keras_convert(model, path)
    #models = Model(inputs=keras_model.input, outputs=keras_model.output)
    #pred = keras_model.predict(image)
    #print(pred)
