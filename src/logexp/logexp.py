import numpy as np
import os
import torch
import shutil
import sys
from pathlib import Path
import logging
import importlib
from src.plotlies.plotlies import *

###### Log class ##############

def get_logger(ch_log_level=logging.ERROR,
               fh_log_level=logging.INFO):
    logging.shutdown()
    importlib.reload(logging)  # Sostituito imp.reload con importlib.reload
    logger = logging.getLogger("cheatsheet")
    logger.setLevel(logging.DEBUG)

    # Console Handler
    if ch_log_level:
        ch = logging.StreamHandler()
        ch.setLevel(ch_log_level)
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch)

    # File Handler
    if fh_log_level:
        fh = logging.FileHandler('cheatsheet.log')
        fh.setLevel(fh_log_level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
###### Log class ##############

###### Xperiment class ##############
class Experiment():
    def __init__(self, name, root, logger=None):
        self.name = name
        self.root = os.path.join(root, name)
        self.logger = logger
        self.epoch = 1
        self.best_val_loss = sys.maxsize
        self.best_val_loss_epoch = 1
        self.weights_dir = os.path.join(self.root, 'weights')
        self.history_dir = os.path.join(self.root, 'history')
        self.results_dir = os.path.join(self.root, 'results')
        self.latest_weights = os.path.join(self.weights_dir, 'latest_weights.pth')
        self.latest_optimizer = os.path.join(self.weights_dir, 'latest_optim.pth')
        self.best_weights_path = self.latest_weights
        self.best_optimizer_path = self.latest_optimizer
        self.train_history_fpath = os.path.join(self.history_dir, 'train.csv')
        self.val_history_fpath = os.path.join(self.history_dir, 'val.csv')
        self.test_history_fpath = os.path.join(self.history_dir, 'test.csv')
        self.loss_history = {
            'train': np.array([]),
            'val': np.array([]),
            'test': np.array([])
        }
        self.precision_history = {
            'train': np.array([]),
            'val': np.array([]),
            'test': np.array([])
        }

    def log(self, msg):
        if self.logger:
            self.logger.info(msg)

    def init(self):
        self.log("Creating new experiment")
        self.init_dirs()
        self.init_history_files()

    def resume(self, model, optim, weights_fpath=None, optim_path=None):
        self.log("Resuming existing experiment")
        if weights_fpath is None:
            weights_fpath = self.latest_weights
        if optim_path is None:
            optim_path = self.latest_optimizer

        model, state = self.load_weights(model, weights_fpath)
        optim = self.load_optimizer(optim, optim_path)

        self.best_val_loss = state['best_val_loss']
        self.best_val_loss_epoch = state['best_val_loss_epoch']
        self.epoch = state['last_epoch'] + 1
        self.load_history_from_file('train')
        self.load_history_from_file('val')

        return model, optim

    def init_dirs(self):
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.history_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def init_history_files(self):
        Path(self.train_history_fpath).touch()
        Path(self.val_history_fpath).touch()
        Path(self.test_history_fpath).touch()


    def load_history_from_file(self, dset_type):
        fpath = os.path.join(self.history_dir, dset_type + '.csv')
        try:
            data = np.loadtxt(fpath, delimiter=',').reshape(-1, 3)
            self.loss_history[dset_type] = data[:, 1]
            self.precision_history[dset_type] = data[:, 2]
        except:
            self.loss_history[dset_type] = np.array([])
            self.precision_history[dset_type] = np.array([])


    def append_history_to_file(self, dset_type, loss, precision):
        fpath = os.path.join(self.history_dir, dset_type + '.csv')
        with open(fpath, 'a') as f:
            f.write('{},{},{}\n'.format(self.epoch, loss, precision))

    def save_history(self, dset_type, loss, precision):
        self.loss_history[dset_type] = np.append(self.loss_history[dset_type], loss)
        self.precision_history[dset_type] = np.append(self.precision_history[dset_type], precision)
        self.append_history_to_file(dset_type, loss, precision)

        if dset_type == 'val' and self.is_best_loss(loss):
            self.best_val_loss = loss
            self.best_val_loss_epoch = self.epoch

        if dset_type == 'val':
            self.plot_and_save_history()

    def is_best_loss(self, loss):
        return loss < self.best_val_loss

    def save_weights(self, model, trn_loss, val_loss, trn_precision, val_precision):
        weights_fname = self.name + '-weights-%d-%.3f-%.3f-%.3f-%.3f.pth' % (
            self.epoch, trn_loss, trn_precision, val_loss, val_precision)
        weights_fpath = os.path.join(self.weights_dir, weights_fname)
        torch.save({
            'last_epoch': self.epoch,
            'trn_loss': trn_loss,
            'val_loss': val_loss,
            'trn_precision': trn_precision,
            'val_precision': val_precision,
            'best_val_loss': self.best_val_loss,
            'best_val_loss_epoch': self.best_val_loss_epoch,
            'experiment': self.name,
            'state_dict': model.state_dict()
        }, weights_fpath)
        shutil.copyfile(weights_fpath, self.latest_weights)
        if self.is_best_loss(val_loss):
            self.best_weights_path = weights_fpath

    def load_weights(self, model, fpath):
        self.log("loading weights '{}'".format(fpath))
        state = torch.load(fpath,map_location=torch.device(device="cuda" if torch.cuda.is_available() else "cpu"))
        model.load_state_dict(state['state_dict'])
        self.log("loaded weights from experiment %s (last_epoch %d, trn_loss %s, trn_precision %s, val_loss %s, val_precision %s)" % (
            self.name, state['last_epoch'], state['trn_loss'],
            state['trn_precision'], state['val_loss'], state['val_precision']))
        return model, state

    def save_optimizer(self, optimizer, val_loss):
        optim_fname = self.name + '-optim-%d.pth' % (self.epoch)
        optim_fpath = os.path.join(self.weights_dir, optim_fname)
        torch.save({
            'last_epoch': self.epoch,
            'experiment': self.name,
            'state_dict': optimizer.state_dict()
        }, optim_fpath)
        shutil.copyfile(optim_fpath, self.latest_optimizer)
        if self.is_best_loss(val_loss):
            self.best_optimizer_path = optim_fpath

    def load_optimizer(self, optimizer, fpath):
        self.log("loading optimizer '{}'".format(fpath))
        optim = torch.load(fpath,map_location=torch.device(device="cuda" if torch.cuda.is_available() else "cpu"))
        optimizer.load_state_dict(optim['state_dict'])
        self.log("loaded optimizer from session {}, last_epoch {}"
                 .format(optim['experiment'], optim['last_epoch']))
        return optimizer

    def plot_and_save_history(self):
        if not hasattr(self, 'monitor'):
            self.monitor = TrainingMonitor(figsize=(12, 8))

        try:
            if os.path.exists(self.train_history_fpath) and os.path.exists(self.val_history_fpath):
                train_data = np.loadtxt(self.train_history_fpath, delimiter=',')
                val_data = np.loadtxt(self.val_history_fpath, delimiter=',')

                if train_data.ndim == 1:
                    train_data = train_data.reshape(1, -1)
                if val_data.ndim == 1:
                    val_data = val_data.reshape(1, -1)

                latest_epoch = int(train_data[-1, 0])
                train_loss = train_data[-1, 1]
                train_precision = train_data[-1, 2]
                val_loss = val_data[-1, 1]
                val_precision = val_data[-1, 2]

                self.monitor.update(
                    epoch=latest_epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    train_precision=train_precision,
                    val_precision=val_precision
                )

                self.monitor.save(os.path.join(self.history_dir, 'training_progress.png'))

        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error updating plot: {e}")

###### Xperiment class ##############