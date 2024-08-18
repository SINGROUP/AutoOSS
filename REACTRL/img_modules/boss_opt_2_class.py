from REACTRL.img_modules.img_data_2_class import train_loader, test_loader
from REACTRL.env_modules.net_framework import *
from REACTRL.env_modules.img_conv import *
from boss.bo.bo_main import BOMain
from boss.pp.pp_main import PPMain
from datetime import datetime
import pandas as pd

def func(X, model=None, method='resnet18'):
    lr = 10 ** X[0, 0]
    output_dir='output_product_2_%s_%s_%s' % (lr, method, datetime.now().strftime("%Y%m%d-%H%M"))
    if model is None:
        model = ResNet18(1, 2)
    net=img_classifier(model, device='cuda', output_dir=output_dir, learning_rate=lr)
    net.model_train(train_loader=train_loader, val_loader=test_loader, episodes=800, device='cuda', tensorboard=True)
    data=pd.read_csv(os.path.join(output_dir, 'train_output_episode.txt'), sep=',')
    # acc=data[' Accuracy'].max()
    
    # loss=data[' Loss'].min()
    # log_loss=np.log10(loss)
    loss=data[' Loss'].tolist()[-1]
    return loss

bounds = np.array([[-5.5, -2.5]])