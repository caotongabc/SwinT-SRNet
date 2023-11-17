import logging

import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_Acc = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_acc, model, index_iter,number,desc):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model, index_iter,number,desc)
            self.counter = 0

    def save_checkpoint(self, val_acc,model , epoch,number,desc):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_Acc:.6f} --> {val_acc:.6f}).  Saving model ...')
            logging.info(f'Validation loss decreased ({self.val_Acc:.6f} --> {val_acc:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), self.path)
        torch.save(model.state_dict(), "../ellaryStop/weights{}zhe-224/{}.pth" .format(number,desc))
        # torch.save(generator_s.state_dict(), 'weightIter1/weights%d/Generator_ema.pth' % (index_iter + 10))
        print("Save model state.")
        self.val_Acc = val_acc
