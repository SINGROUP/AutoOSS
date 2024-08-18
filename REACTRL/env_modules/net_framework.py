
# Import libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt
import os
import pickle
import numpy as np

from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score



class img_classifier(nn.Module):
    def __init__(self, model, train_loader=None, val_loader=None, output_dir='output', learning_rate=0.0001, tensorboard=True, epochs=200, optimizer='Adam', scheduler='StepLR', loss_fn='cross_entropy_loss', device="cpu"):
        super(img_classifier, self).__init__()

        self.model = model
        self.model = self.model_to_device(device)


        if optimizer=='Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer=='SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
            
        if scheduler=='StepLR':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        elif scheduler=='ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        elif scheduler=='CosineAnnealingLR':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=0)
            
            
            
        if loss_fn=='cross_entropy_loss':
            self.loss_fn = nn.CrossEntropyLoss()

        if tensorboard:
            self.tensorboard_writer = SummaryWriter()


        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self.output_dir = output_dir



        with open(os.path.join(output_dir, "train_output_batch.txt"), "a") as train_output_batch:
            train_output_batch.write("Training output batch\n")
        self.train_output_batch = train_output_batch

        with open(os.path.join(output_dir, "train_output_episode.txt"), "a") as train_output_episode:
            train_output_episode.write("Episode, Loss, Accuracy\n")
        self.train_output_episode = train_output_episode
        
        with open(os.path.join(output_dir, "val_output_batch.txt"), "a") as val_output_batch:
            val_output_batch.write("Validation output batch\n")
        self.val_output_batch = val_output_batch
        with open(os.path.join(output_dir, "val_output_episode.txt"), "a") as val_output_episode:
            val_output_episode.write("Episode, Loss, Accuracy\n")
        self.val_output_episode = val_output_episode

        with open(os.path.join(output_dir, "test_output_batch.txt"), "a") as test_output_batch:
            test_output_batch.write("Test output batch\n")
        self.test_output_batch = test_output_batch
        with open(os.path.join(output_dir, "test_output_episode.txt"), "a") as test_output_episode:
            test_output_episode.write("Episode, Loss, Accuracy\n")
        self.test_output_episode = test_output_episode
    

    def model_train(self,
                    episodes=200,
                    optimizer=None,
                    scheduler=None, 
                    loss_fn=None, 
                    train_loader=None, 
                    val_loader=None,  
                    device=None,
                    tensorboard=None,):
        """
        Trains the model on the given dataset for the specified number of epochs. 
        Parameters
        ----------
        model: torch.nn.Module
            The neural network model to be trained
        optimizer: torch.optim.Optimizer
            The optimizer to be used for training
        loss_fn: torch.nn.modules.loss
            The loss function to be used for training
        train_loader: torch.utils.data.DataLoader
            The training dataset
        val_loader: torch.utils.data.DataLoader
            The validation dataset
        epochs: int
            The number of epochs for which the model is to be trained
        device: str
            The device on which the model is to be trained
        Returns
        -------
        model: torch.nn.Module
            The trained model
        train_losses: list
            The training losses for each epoch
        val_losses: list
            The validation losses for each epoch
        """
        
        if optimizer is None:
            optimizer = self.optimizer
        if scheduler is None:
            scheduler = self.scheduler
        if loss_fn is None:
            loss_fn = self.loss_fn
        if train_loader is None:
            train_loader = self.train_loader
        if val_loader is None:
            val_loader = self.val_loader
        if device is None:
            device = self.device
        if tensorboard is None:
            tensorboard = self.tensorboard
        
    
            


        loss_min=10000000
        for episode in range(episodes):
            self.model.train()
            acc_episode=[]
            loss_episode=[]
            
            for batch, (X, y) in enumerate(train_loader):
                inputs=torch.unsqueeze(X, 1).float().to(device)  # img
                # inputs=torch.permute(X, (0, 2, 1)).float().to(device)  # current cnn

                # inputs=X.float().to(device) # signal topography
                
                # inputs=torch.permute(X, (0, 3, 1, 2)).float()

                labels=y.float().to(device)
                optimizer.zero_grad()
            
                preds = self.model(inputs)
                loss = self.loss_fn(preds, labels)
                acc=(torch.max(F.softmax(preds, dim=1), dim = 1).indices==torch.max(F.softmax(labels, dim=1), dim = 1).indices).sum()
                # print('labels:', labels, 'preds:', preds)
                loss.backward()
                optimizer.step()
                loss_episode.append(loss.item())
                acc_episode.append(acc.item())
                with open(os.path.join(self.output_dir, "train_output_batch.txt"), "a") as train_output_batch:
                    train_output_batch.write(f"Episode: {episode}, Batch: {batch}, Loss: {loss.item()}, Accuracy: {acc.item()}\n")
                if tensorboard:
                    self.tensorboard_writer.add_scalar('Loss/train', loss.item(), episode*len(train_loader) + batch)
                    self.tensorboard_writer.add_scalar('Accuracy/train', acc.item(), episode*len(train_loader) + batch)

            scheduler.step()
            loss_episode_avg = np.mean(loss_episode)
            acc_episode = np.sum(acc_episode)/len(train_loader.dataset)
            with open(os.path.join(self.output_dir, "train_output_episode.txt"), "a") as train_output_episode:
                train_output_episode.write(f"{episode}, {loss_episode_avg}, {acc_episode}\n")
            if tensorboard:
                self.tensorboard_writer.add_scalar('Loss_episode/train', loss_episode_avg, episode)
                self.tensorboard_writer.add_scalar('Accuracy_episode/train', acc_episode, episode)
                

            if episode%10==0:
                # validation loss
                val_loss, val_acc = self.model_test(val_loader, mode='val', device=device, label=episode)
                with open(os.path.join(self.output_dir, "val_output_episode.txt"), "a") as val_output_episode:
                    val_output_episode.write(f"{episode}, {val_loss}, {val_acc}\n")
                if tensorboard:
                    self.tensorboard_writer.add_scalar('Loss_episode/val', val_loss, episode)
                    self.tensorboard_writer.add_scalar('Accuracy_episode/val', val_acc, episode)
                # self.model_save(os.path.join(self.output_dir, f"model_{episode}.pth"))
                if val_loss<loss_min:
                    loss_min=val_loss
                    self.model_save(os.path.join(self.output_dir, "model_best.pth"))
                    
            if episode==episodes-1:
                self.model_save(os.path.join(self.output_dir, "model_last.pth"))
                

    


    
    def model_test(self,
                test_loader,
                mode="test",
                label=None, 
                device="cpu"):
        """
        Tests the model on the given dataset. 
        Parameters
        ----------
        model: torch.nn.Module
            The neural network model to be trained
        test_loader: torch.utils.data.DataLoader
            The test dataset
        device: str
            The device on which the model is to be trained
        Returns
        -------
        test_loss: float
            The test loss
        test_accuracy: float
            The test accuracy
        """
        self.model.eval()
        acc_episode=[]
        loss_episode=[]
        with torch.no_grad():
            for batch, (X, y) in enumerate(test_loader):
                inputs=torch.unsqueeze(X, 1).float().to(device)
                # inputs=torch.permute(X, (0, 3, 1, 2)).float()
                # inputs=torch.permute(X, (0, 2, 1)).float().to(device) # current
                labels=y.float().to(device)
            
                preds = self.model(inputs)
                loss = self.loss_fn(preds, labels)
                acc=(torch.max(F.softmax(preds, dim=1), dim = 1).indices==torch.max(F.softmax(labels, dim=1), dim = 1).indices).sum()
                loss_episode.append(loss.item())
                acc_episode.append(acc.item())
                if mode=="test":
                    with open(os.path.join(self.output_dir, "test_output_batch.txt"), "a") as test_output_batch:
                        test_output_batch.write(f"Episode: {label}, Batch: {batch}, Loss: {loss.item()}, Accuracy: {acc.item()}\n")
                elif mode=="val":
                    with open(os.path.join(self.output_dir, "val_output_batch.txt"), "a") as val_output_batch:
                        val_output_batch.write(f"Episode: {label}, Batch: {batch}, Loss: {loss.item()}, Accuracy: {acc.item()}\n")

        loss_episode_avg = np.mean(loss_episode)
        acc_episode = np.sum(acc_episode)/len(test_loader.dataset)
        return loss_episode_avg, acc_episode
    
    def model_predict(self, inputs):
        """
        Predict the output for the given images
        Parameters
        ----------
        inputs: cv2 images, default channel with 1
        Returns
        -------
        Returns the predicted output
        """    
        
        self.model.eval()
        with torch.no_grad():
            preds = self.model(inputs)
        return preds
    
    def model_save(self, 
                path):
        """
        Saves the model at the specified path
        Parameters
        ----------
        path: str
            The path at which the model is to be saved
        Returns
        -------
        None
        """
        torch.save(self.model.state_dict(), path)

    def model_load(self, weight_path):
        self.model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))

            


    def calculate_auc_roc(self, preds, labels, pos_label=None, plot_graph=True):
        fpr, tpr, thresh = roc_curve(preds, labels, pos_label=pos_label)
        auc_score = roc_auc_score(preds, labels)
        if plot_graph:
            plt.figure(figsize=(3, 3))
            fig, ax = plt.subplots()
            plt.setp(ax.spines.values(), linewidth=3)
            # The ticks
            ax.xaxis.set_tick_params(width=3)
            ax.yaxis.set_tick_params(width=3)
            plt.plot(fpr, tpr)
            plt.ylabel('True Positive Rate', fontsize=16)
            plt.xlabel('False Positive Rate', fontsize=16)
            plt.title('Receiver operating characteristic', fontsize=16)
            plt.text(0.5, 0.5, f'AUC score: %.3f' % auc_score, fontsize=14)
            plt.savefig('ROC.png')
        return fpr, tpr, thresh, auc_score
    
    def calculate_accuracy(self, preds, labels):
        return (torch.max(F.softmax(preds, dim=1), dim = 1).indices==torch.max(F.softmax(labels, dim=1), dim = 1).indices).sum()
    
    def get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def model_to_device(self, device):
        if isinstance(self.model, (list,tuple)):
            return [self.to_device(x, device) for x in self.model]
        return self.model.to(device, non_blocking=True)
    


