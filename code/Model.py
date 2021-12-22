import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from NetWork import ResNet
from ImageUtils import parse_record, preprocess_private_test

""" This script defines the training, validation and testing process.
"""

class Cifar(nn.Module):
    def __init__(self, config):
        super(Cifar, self).__init__()
        self.config = config
        self.network = ResNet(
            self.config.resnet_size,
            self.config.num_classes,
            self.config.first_num_filters,
        )
        
        # Cross entropy loss and Optimizer
        self.loss = nn.CrossEntropyLoss()
        self.lr = 0.01
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.lr, momentum=0.9, weight_decay = self.config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        
    # The epoch checkpoint loads a saved epoch and continues training from that epoch
    # Implements the training for given dataset for max_epochs number of epochs
    def train(self, x_train, y_train, max_epoch, epoch_ckpt):
        if(epoch_ckpt != 0):
            checkpointfile = os.path.join(self.config.checkpoints_dir, 'model-%d.ckpt'%(epoch_ckpt))
            checkpoint = torch.load(checkpointfile)
            self.network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
        else:
            self.epoch = 1
        self.network.train()

        # Determine how many batches in an epoch
        num_samples = x_train.shape[0]
        num_batches = num_samples // self.config.batch_size

        print('### Training... ###')
        for epoch in range(self.epoch, self.epoch + max_epoch+1):
            start_time = time.time()
            # Shuffle the dataset
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]

            self.network = self.network.cuda()
            
            
            for i in range(num_batches):
                
                # Construct the current batch.
                curr_x_batch = []
                for j in range(i*self.config.batch_size , (i+1)*self.config.batch_size):
                    curr_x_batch.append(parse_record(curr_x_train[j], True))

                curr_y_batch = curr_y_train[i*self.config.batch_size : (i+1)*self.config.batch_size]

                curr_y_batch = np.array(curr_y_batch)

                curr_x_batch_tensor = torch.stack(curr_x_batch).float().cuda()
                curr_y_batch_tensor = torch.tensor(curr_y_batch).float().cuda()

                self.model = self.network.cuda()
                output = self.model(curr_x_batch_tensor)

                loss = self.loss(output, curr_y_batch_tensor.long())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss), end='\r', flush=True)
            
            # Uses cosine annealing to change learning rate
            self.scheduler.step()
            
            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, loss, duration))

            if epoch % self.config.save_interval == 0 or (epoch > 150 and epoch % 5 == 0) or (epoch > 180 and loss < 0.001):
                self.save(epoch, loss)


    def evaluate(self, x, y, checkpoint_num_list):
        
        print('### Test or Validation ###')
        for checkpoint_num in checkpoint_num_list:
            checkpointfile = checkpointfile = os.path.join(self.config.checkpoints_dir, 'model-%d.ckpt'%(checkpoint_num))
            checkpoint = torch.load(checkpointfile)
            self.network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']

            self.network.eval()

            preds = []
            curr_x = []
            for i in tqdm(range(x.shape[0])):   
                curr_x = parse_record(x[i], False, False)

                curr_x_tensor = curr_x.float().cuda().view(1,3,32,32)

                preds.append(int(torch.max(self.network(curr_x_tensor),1)[1]))

            y = torch.tensor(y)
            preds = torch.tensor(preds)
            accuracy = torch.sum(preds==y)/y.shape[0]
            print('Test accuracy: {:.4f}'.format(accuracy), 'for model-%d.ckpt'%(checkpoint_num))
    
    def save(self, epoch, loss):
        checkpoint_path = os.path.join(self.config.checkpoints_dir, 'model-%d.ckpt'%(epoch))
        os.makedirs(self.config.checkpoints_dir, exist_ok=True)
        # Save the model details, epoch number, optimmizer - This allows load from ckpt and train
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            }, checkpoint_path)

        print("Checkpoint has been created.")
    
    # Load the checkpoint and get network weights
    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))

    # Stores probabilities for prediction on private test data
    def predict_prob(self, x, checkpoint_num):
        
        print('### Prediction ###')
        checkpointfile = os.path.join(self.config.checkpoints_dir, 'model-%d.ckpt'%(checkpoint_num))
        # checkpointfile = os.path.join('/content/drive/MyDrive/Colab Notebooks/model-180.ckpt')
        checkpoint = torch.load(checkpointfile)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        loss = checkpoint['loss']


        self.network.eval()
        
        preds = []
        for i in tqdm(range(x.shape[0])):   
            curr_x = preprocess_private_test(x[i])

            curr_x_tensor = torch.tensor(curr_x).float().cuda().view(1,3,32,32)
            
            pred = self.network(curr_x_tensor)
            pred_np = torch.nn.functional.softmax(pred).cpu().detach().numpy().reshape((10))
            # print(pred_np)
            preds.append(pred_np)

        preds_np = np.array(preds)
        print(preds_np.shape)
        np.save('/content/drive/MyDrive/Colab Notebooks/Project/predictions.npy', preds_np)
