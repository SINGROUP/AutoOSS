
from REACTRL.env_modules.net_framework import *
from REACTRL.env_modules.signal_conv import *
from REACTRL.img_modules.signal_data import train_loader, test_loader
from boss.bo.bo_main import BOMain
from boss.pp.pp_main import PPMain
from datetime import datetime
import pandas as pd

def func(X, model=None, method='lstm'):
    lr = 10 ** X[0, 0]
    output_dir='output_%s_%s_%s' % (lr, method, datetime.now().strftime("%Y%m%d-%H%M"))
    if model is None:
        input_size = 1
        hidden_size = 50
        num_classes = 2  # Adjust based on your classification task

        # Instantiate the model
        # model = TimeSeriesClassifier(input_size, hidden_size, num_classes)
        model = LSTMClassifier(1, 256, 3, 2)
    net=img_classifier(model, device='cuda', output_dir=output_dir, learning_rate=lr)
    net.model_train(train_loader=train_loader, val_loader=test_loader, episodes=1000, device='cuda', tensorboard=True)
    data=pd.read_csv(os.path.join(output_dir, 'train_output_episode.txt'), sep=',')
    # acc=data[' Accuracy'].max()
    
    # loss=data[' Loss'].min()
    # log_loss=np.log10(loss)
    loss=data[' Loss'].tolist()[-1]
    return loss

bounds = np.array([[-6, -2]])