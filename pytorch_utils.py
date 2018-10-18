import torch
import torch.nn as nn
import numpy as np
import datetime

from collections import OrderedDict
from helperFunctions.printing import print_color


def normal_init(param, mu=0, std=1):
    nn.init.normal(param, mu, std)


def xavier(param):
    print('test')
    nn.init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Embedding) or isinstance(m, nn.Linear):
        xavier(m.weight.data)
    if isinstance(m, nn.Linear):
        m.bias.data.normal_(0, 1)


def save_weights(net, args, iteration=None, final=False):
    """
    :param net: network for which to save the weights
    :param args:    either a namedtuple or arguments from parse args. This must include weight_dir,
                    embed_dim, etc.
    :param iteration: the iteration number the weights will be saved after
    :param final: are these the final weights after full training?
    :return: the file name the weights are saved to
    """

    # TODO: probably not general enough to be inlcuded in the pytorch utils
    # TODO: maybe if I call this save_embedding_weights?

    if final:
        time_stamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        unique_id = '_{}'.format(time_stamp)
        out_name = args.weight_dir + args.out_file + \
                   'final_{}embdim_{}fix_{}lr_{}'.format(
                       args.embed_dim, args.subsample_fixtures, args.lr, args.sparse) \
                   + unique_id + '.pth'
        print_color('cyan', 'Saving final weights', out_name)
        torch.save(net.state_dict(), out_name)
    else:
        out_name = args.weight_dir + args.out_file + \
                   'iter{}_{}embdim_{}fix_{}lr_{}'.format(
                       iteration, args.embed_dim, args.subsample_fixtures,
                       args.subsample_fixtures, args.lr, args.sparse) + '.pth'
        print_color('cyan', 'Saving state, iter:', iteration, out_name)
        torch.save(net.state_dict(), out_name)

    # save oufile name to text file for future reference
    with open(args.ref_dir + args.ref_file, 'a') as f:
        f.write(out_name + '\n')

    return out_name


def save_weights2(net, iteration=None, out_name='testfile'):

    out_name = out_name + '{}.pth'.format(iteration)
    print_color('cyan', 'Saving state, iter:', iteration, out_name)
    torch.save(net.state_dict(), out_name)


def prepare_state_dict(weight_file, saved_on_gpu, use_cuda):

    """
    Function to alter the pytorch state dictionary if changing between using gpu and cpu. If not
    the state dictionary is left unaltered.
    :param weight_file:
    :param saved_on_gpu: Whether or not the training was done on a gpu
    :param use_cuda:  Whether to use gpu currently
    :return: Returns the possibly altered state dictionary
    """

    if saved_on_gpu and not use_cuda:
        state_dict = torch.load(weight_file, map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove module from names (module is only used in gpu training and infer)
        new_state_dict[name] = v
        return new_state_dict
    # elif not saved_on_gpu and not use_cuda:
    else:
        return torch.load(weight_file, map_location='cpu')
        # TODO: what if we are doing this on gpu -- need two other options here


def add_to_weight_matrix(state_dict, layer_name, m, n=0, initialization_function=xavier):
    """
    A utility function for adding more weights to a previously trained network. This may be useful
    for user embedding if a new user is seen but can be used on a general network also when complete
    network retraining is impractical or even impossible.
    :param state_dict: the state dictionary containing previously trained weights
    :param layer_name: the name of the network layer to be altered
    :param m: the number of rows (number of layer inputs) to be added to the state dictionary
    :param n: the number of columns (number of layer outputs) to be added to the state dictionary
    :param initialization_function: function to initialize the newly added weights
    :return: new state dictionary with larger weight matrix for specified layer
    """
    initial_weight_matrix = state_dict[layer_name]
    initial_weight_dim0 = initial_weight_matrix.shape[0]
    initial_weight_dim1 = initial_weight_matrix.shape[1]
    # initialize larger weight matrix with the correct shape
    weights = torch.Tensor(np.zeros((initial_weight_dim0 + m, initial_weight_dim1 + n)))
    initialization_function(weights)
    # transfer current weights to new weight matrix
    weights[:-m, :-n] = initial_weight_matrix
    weights = torch.Tensor(weights)
    # rewrite the layer weights in the state dictionary with new ones.
    state_dict[layer_name] = weights
    return state_dict
