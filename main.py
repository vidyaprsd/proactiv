import torchvision
import time
import os
from torch.utils.data import DataLoader
import itertools
import torch
from torch import nn as nn
import numpy as np
from spinalvgg import SpinalVGG
from data_loader import ImageDatasetFromPt
from trainable_tensor import TrainableTensor
from utils import save_data
from matplotlib import pyplot as plt

'''
This Script contains the main training script for the input optimization.
Author: @vidyaprsd
'''

if __name__ == '__main__':
    batch_size_test         = 520
    numclasses              = 26
    learning_rate           = 0.001   # Set by experimentation on 1-2 sample images.
    cutoff_loss             = 0.01    # #Min loss beyond which input optimization is stopped.
    num_iters               = 500     # #Max iterations for the input optimization process.
    cuda                    = True
    plotinterim_checkpoint  = 50      # Set to -1 to turn off.

    # Downloaded from here https://www.nist.gov/itl/products-and-services/emnist-dataset
    data_path               = './test_letters.pt'

    # Trained based on code available from authors of SpinaVGG.
    # https://github.com/dipuk0506/SpinalNet/blob/master/MNIST_VGG/EMNIST_letters_VGG_and%20_SpinalVGG.py
    pretrained_model_path   = './model.pth.tar'

    # Same loss used for the model training.
    criterion               = nn.CrossEntropyLoss()

    # Transforms to be tested.
    transforms              = {'types': ['angle', 'blur', 'noise_level'],
                               'values': list(itertools.product([0, -15, -12, -10, -8, -6, -4, -2, 2, 4, 6, 8, 10, 12, 15],
                                                                [0, 1, 2],
                                                                [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))}

    logdir                  = 'log_' + str(time.time())  # Output directory
    device                  = 'cuda' if cuda else 'cpu'

    #Normalize inputs
    preproc                 = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                              torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    #load pretrained model. We do not change the weights of these.
    model                   = SpinalVGG().to(device)
    checkpoint              = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    #iterate over transforms T
    for transform in transforms['values']:
        print(transform)

        log_subdir          = os.path.join(logdir, '_'.join(map(str, transform)))
        if not os.path.exists(log_subdir):
            os.makedirs(log_subdir)

        #initialize the dataset loader params to generate images changed by $transform value
        dataset_params      = {'data_path': data_path, 'preproc': preproc, 'cnt': int(batch_size_test / numclasses), 'cuda': cuda}
        for i in range(len(transform)):
            dataset_params[transforms['types'][i]] = transform[i]
        dataset             = ImageDatasetFromPt(**dataset_params)
        data_loader         = DataLoader(dataset=dataset, batch_size=batch_size_test, shuffle=False, num_workers=0, pin_memory=False)

        #iterate over all images changed by $transform
        for index, batch in enumerate(data_loader):
            print(index)
            outputs = model(batch['input'])
            loss = criterion(outputs, batch['target'])

            save_data(log_subdir, index, batch['input'], outputs, batch['target'],'act')

            #initialize a trainable input tensor with the input images, i.e., batch['input']
            trainable_input = TrainableTensor(batch['input'])

            # Initialize the optimizer parameters with the parameters of the trainable input tensor.
            # Usually in model training, the optimizer parameters is set to model.parameters(). Here,
            # We change that to trainable_input.parameters() and to update the input while keeping the model constant.
            optimizer = torch.optim.Adam(trainable_input.parameters(), lr=learning_rate)
            prev_loss = 9999

            for epoch in range(num_iters):
                #The trainable input is passed as a model input and the corresponding loss computed.
                outputs = model(trainable_input.tensor)
                _, predict = torch.max(outputs.data, 1)
                loss = criterion(outputs, batch['target'])

                optimizer.zero_grad() # Reset gradients to 0
                loss.backward() # Recompute gradients via backpropagation w.r.t the trainable_input parameters, i.e., input image pixels.
                optimizer.step() # Update the trainable_input parameters/pixels.
                prev_loss = loss

                if loss < cutoff_loss: #break if the minimum loss is reached.
                    break
                print("Batch: ", index, "; Epoch: ", epoch, "; Loss: ",np.round(loss.item(), 4))

                if plotinterim_checkpoint > 0 and (epoch%plotinterim_checkpoint ==0 or epoch == num_iters-1) :
                    for imgid in range(batch['input'].shape[0]):
                        fig, ax = plt.subplots(1, 2)
                        ax[0].imshow(batch['input'][imgid, 0, :, :].data.cpu().numpy(), cmap='gray')
                        ax[0].set_title('Actual Input')
                        ax[1].imshow(trainable_input.tensor[imgid, 0, :, :].data.cpu().numpy(), cmap='gray')
                        ax[1].set_title('Optimized Input')
                        fig.suptitle("Batch: "+ str(index)+ "; Epoch: "+ str(epoch))
                        plt.show()
                        break

            outputs = model(trainable_input.tensor)
            save_data(log_subdir, index, trainable_input.tensor, outputs, batch['target'], 'opt')

