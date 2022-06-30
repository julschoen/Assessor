import os
import numpy as np
import pickle

import torch 
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable, grad
from torch.cuda.amp import autocast, GradScaler

from temp_disc_alt import Discriminator as AssesorRes
from temp_disc import Discriminator as Assesor


class Trainer(object):
    def __init__(self, dataset, dataset_val, params):
        ### Misc ###
        self.p = params
        self.device = params.device

        ### Make Dirs ###
        self.log_dir = params.log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.models_dir = os.path.join(self.log_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)

        ### load/save params
        if params.load_params:
            with open(os.path.join(params.log_dir, 'params.pkl'), 'rb') as file:
                params = pickle.load(file)
        else:
            with open(os.path.join(params.log_dir,'params.pkl'), 'wb') as file:
                pickle.dump(params, file)

        ### Make Models ###
        if self.p.res:
            self.assesor = AssesorRes(self.p).to(self.device)
        else:
            self.assesor = Assesor(self.p).to(self.device)
        
        if self.p.ngpu>1:
            self.assesor = nn.DataParallel(self.assesor)
            

        self.opt = optim.Adam(self.assesor.parameters(), lr=self.p.lr, betas=(0., 0.9))
        self.scaler = GradScaler()

        ### Make Data Generator ###
        self.generator_train = DataLoader(dataset, batch_size=self.p.batch_size, shuffle=True, num_workers=4, drop_last=True)
        self.val_data = DataLoader(dataset_val, batch_size=self.p.batch_size, shuffle=True, num_workers=4)
        self.val_length = dataset_val.__len__()

        ### Prep Training
        self.losses = []
        self.val_losses = []

        self.cla_loss = nn.BCEWithLogitsLoss()

    def inf_train_gen(self):
        while True:
            for data in self.generator_train:
                yield data
        
    def log_train(self, step):
        l = self.losses[-1]
        vl, acc = self.val_losses[-1]

        print('[%d/%d] Loss: %.2f\tVal Loss: %.2f Acc: %.2f' % (step,self.p.niters,l,vl, acc))

    def start_from_checkpoint(self):
        step = 0
        files = [f for f in os.listdir(self.models_dir)]
        if len(files) < 2:
            checkpoint = os.path.join(self.models_dir, 'checkpoint.pt')
        else:
            files.remove('checkpoint.pt')
            files = sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]))
            checkpoint = os.path.join(self.models_dir, files[-1])
        if os.path.isfile(checkpoint):
            state_dict = torch.load(checkpoint)
            step = state_dict['step']

            self.assesor.load_state_dict(state_dict['model'])
            self.opt.load_state_dict(state_dict['opt'])
            self.losses = state_dict['loss']
            self.val_losses = state_dict['val_loss']

            print('starting from step {}'.format(step))
        return step

    def save_checkpoint(self, step):
        if step < self.p.niters - 1001:
            name = 'checkpoint.pt'
        else:
            name = f'checkpoint_{step}.pt'

        torch.save({
        'step': step,
        'model': self.assesor.state_dict(),
        'opt': self.opt.state_dict(),
        'loss': self.losses,
        'val_loss': self.val_losses,
        }, os.path.join(self.models_dir, name))

    def log(self, step):
        if step % self.p.steps_per_log == 0:
            self.step_val()
            self.log_train(step)

    def log_final(self, step):
        self.log_train(step)
        self.save_checkpoint(step)

    def step_train(self, x,y):
        for p in self.assesor.parameters():
            p.requires_grad = True
        
        self.assesor.zero_grad()
        with autocast():
            pred = self.assesor(x)
            err = self.cla_loss(pred,y)

        self.scaler.scale(err).backward()
        self.scaler.step(self.opt)
        self.scaler.update()

        for p in self.assesor.parameters():
            p.requires_grad = False

        return err.item()

    def step_val(self):
        errs = []
        acc = 0
        for _, (x,y) in enumerate(self.val_data):
            with autocast():
                pred = self.assesor(x)
                y = y.to(self.device)
                err = self.cla_loss(pred,y)
                errs.append(err.item())
                acc = acc + torch.sum((pred > 0) == y).item()

        acc = acc/self.val_length
        self.val_losses.append((np.mean(errs), acc))

    def train(self):
        step_done = self.start_from_checkpoint()
        gen = self.inf_train_gen()

        print("Starting Training...")
        for i in range(step_done, self.p.niters):
            x, y = next(gen)
            x, y = x.to(self.device), y.to(self.device)
            err = self.step_train(x,y)

            self.losses.append(err)
            self.log(i)
            if i%100 == 0 and i>0:
                self.save_checkpoint(i)

            
        self.log_final(i)
        print('...Done')

