import itertools
from torch import nn as nn
import time
import os
from utils import mse, is_class_correct, is_class_different, cross_entropy

params = {
    "batch_size_test"           : 520,
    "numclasses"                : 26,
    "learning_rate"             : 0.001,   # Set by experimentation on 1-2 sample images.
    "cutoff_loss"               : 0.05,    # #Min loss beyond which input optimization is stopped.
    "num_iters"                 : 500,     # #Max iterations for the input optimization process.
    "cuda"                      : True,
    "plotinterim_checkpoint"    : -1,      # Set to -1 to turn off.
    "logdir"                    : 'outputs',# + '_' + str(time.time()),  # Output directory


    "pretrained_model_path"     : os.path.join("model", "model.pth.tar"), #pretrained SpinalVGG net. https://github.com/dipuk0506/SpinalNet/blob/master/MNIST_VGG/EMNIST_letters_VGG_and%20_SpinalVGG.py
    "criterion"                 : nn.CrossEntropyLoss(), #loss for input optimiztion. Usually it is the same as the loss function used for model training

    "data_path"                 : os.path.join("data", "test_letters.pt"), # Downloaded from here https://www.nist.gov/itl/products-and-services/emnist-dataset
    "transforms"                : {
        "types"                 : ['angle', 'blur', 'noise_level'], #transform functions T
        "values"                : #corresponding parameters for each transform T
            list(itertools.product([0, -15, -12, -10, -8, -6, -4, -2, 2, 4, 6, 8, 10, 12, 15], #rotations
                                                         [2, 1, 0], #blur_sigma
                                                         [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] #noise levels
                                          ))},

    "differences_fn"              : 'differences.p', #filename for dump of distances d_x, d_y, d_g, d^_g per input x^T_i

    # compute:
    # 0: difference between inputs (d_x),
    # 1: difference between predicted output classes (d_y),
    # 2: difference between transformed output probability and target (d_g),
    # 3: difference between projected output probability and target (d^_g)
    #-1: change in performance (d_g - d^_g)

    "differences_metrics"          :  { #specify distances to be computed
                                    'mse'   : {'function' : mse                , 'compute': [0, 1]}, #computes mse_x, mse_y
                                    'diff'  : {'function' : is_class_different , 'compute': [1]},
                                    'match' : {'function' : is_class_correct   , 'compute': [2, 3, -1]},
                                    'loss'  : {'function' : cross_entropy      , 'compute': [2, 3, -1]},}
}

def get_params():
    return params