"""
The class wrapper for the networks
"""
# Built-in
import os
import time

# Torch
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# Own module
import network_utils


class Network(object):
    def __init__(self, features, labels, model_fn, flags, train_loader, test_loader,
                 ckpt_dir = os.path.join(os.path.abspath(' '), 'models')):
        self.features = features                    # The
        self.labels = labels
        self.model_fn = model_fn
        self.flags = flags
        self.ckpt_dir = os.path.join(ckpt_dir, time.strftime('%Y%m%d_%H%M%S', time.localtime()))
        self.model = self.creat_model()
        self.loss = self.make_loss()
        self.optm = self.make_optimizer()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.log = SummaryWriter(self.ckpt_dir) # Create a summary writer for keeping the summary to the tensor board

    def creat_model(self):
        """
        Function to create the network module from provided model fn and flags
        :return: the created nn module
        """
        return self.model_fn(self.flags)

    def make_loss(self, logit=None, labels=None):
        """
        Create a tensor that represents the loss. This is consistant both at training time \
        and inference time for Backward model
        :param logit: The output of the network
        :return: the total loss
        """
        if logit is None:
            return None
        MSE_loss = nn.functional.mse_loss(logit, labels)          # The MSE Loss of the
        BDY_loss = 0 #Implemenation later
        return MSE_loss + BDY_loss

    def make_optimizer(self):
        """
        Make the corresponding optimizer from the flags. Only below optimizers are allowed. Welcome to add more
        :return:
        """
        if self.flags.optim == 'Adam':
            op = torch.optim.Adam(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'RMSprop':
            op = torch.optim.RMSprop(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'SGD':
            op = torch.optim.SGD(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        else:
            raise Exception("Your Optimizer is neither Adam, RMSprop or SGD, please change in param or contact Ben")
        return op

    def save(self):
        torch.save(self.model.state_dict, self.ckpt_dir)

    def load(self):
        self.model.load_state_dict(torch.load(self.ckpt_dir))

    def train(self):
        """
        The major training function. This would start the training using information given in the flags
        :return: None
        """
        best_validation_loss = 1e-2     # Set a relatively large staring best_val loss
        for epoch in range(self.flags.train_step):
            # Set to Training Mode
            train_loss = 0
            self.model.train()
            for j, (geometry, spectra) in enumerate(self.train_loader):
                geometry = geometry.cuda()                          # Put data onto GPU
                spectra = spectra.cuda()                            # Put data onto GPU
                self.optm.zero_grad()                               # Zero the gradient first
                logit = self.model(geometry)                        # Get the output
                loss = self.make_loss(logit, spectra)              # Get the loss tensor
                self.loss.backward()                                # Calculate the backward gradients
                self.optm.step()                                    # Move one step the optimizer
                train_loss += loss

            if epoch % self.flags.eval_step:
                # Record the training loss to the tensorboard
                train_avg_loss = train_loss / (j+1)
                self.log.add_scalar('Loss/train', train_avg_loss, epoch)

                # Set to Evaluation Mode
                self.model.eval()
                print("Doing Evaluation on the model now")
                test_loss = 0
                for j, (geometry, spectra) in enumerate(self.test_loader):
                    geometry = geometry.cuda()
                    spectra = spectra.cuda()
                    logit = self.model(geometry)
                    loss = self.make_loss(logit, spectra)
                    test_loss += loss

                # Record the testing loss to the tensorboard
                test_avg_loss = test_loss / (j+1)
                self.log.add_scalar('Loss/test', test_avg_loss, epoch)

                print("This is Epoch %d, training loss %.3f, validation loss %.3f" \
                      % (epoch, train_avg_loss, test_avg_loss ))

                # Model improving, save the model down
                if test_avg_loss < best_validation_loss:
                    best_validation_loss = test_avg_loss
                    self.save()
                    print("Saving the model down...")

                    if best_validation_loss < self.flags.stop_threshold:
                        print("Training finished EARLIER at epoch %d, reaching loss of %.3f" %\
                              (epoch, best_validation_loss))
                        return None

