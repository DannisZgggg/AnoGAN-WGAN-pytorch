""" Options

This script is largely based on junyanz/pytorch-CycleGAN-and-pix2pix.

Returns:
    [argparse]: Class containing argparse
"""

import argparse
import os
import torch

# pylint: disable=C0103,C0301,R0903,W0622

class Options():
    """Options class

    Returns:
        [argparse]: argparse containing train and test options
    """

    def __init__(self):
        ##
        #
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        ##
        # Base
        self.parser.add_argument('--category', type=str, default='zero', help='category')
        self.parser.add_argument('--experiment_name', type=str,default='anogan_wgan') #

        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--batchsize', type=int, default=8, help='input batch size') #8
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam') #0.001
        self.parser.add_argument('--niter', type=int, default=500, help='number of epochs to train for')
        self.parser.add_argument('--experiment_group', type=str, default='output_mnist')


    def parse(self):
        """ Parse Arguments.
        """

        self.opt = self.parser.parse_args()
        # self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        # set gpu ids
        torch.cuda.set_device(self.opt.gpu_ids[0])

        self.opt.experiment_name = self.opt.category + '_' + self.opt.experiment_name
        args = vars(self.opt)

        print('Now running: ',self.opt.experiment_name)

        expr_dir = os.path.join(self.opt.experiment_group, self.opt.experiment_name, 'train') #<<TODO
        test_dir = os.path.join(self.opt.experiment_group, self.opt.experiment_name, 'test' )

        if not os.path.isdir(expr_dir):
            os.makedirs(expr_dir)
        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)

        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
