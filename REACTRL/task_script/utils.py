from collections import namedtuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import cv2

from REACTRL    import mol_property
import numpy as np
import shutil
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import pickle


def performance_model_succ_rate(current_time=None, episode_start=0):

    data_diss_all_label=pd.read_csv('c:\\Users\\wun2\\github\\clean_reaction_rl_1209\\diss_all_output\\save_diss_image_all_sac_%s\\all_tip_pos_data_label_%s.txt' % (current_time, current_time), sep='\t', engine='python', skipinitialspace=True)
    data_diss_all_label.rename(columns=lambda x: x.strip(), inplace=True)
    data_diss_all_label['done_diss']=data_diss_all_label['done_diss'].str.replace(' ', '')


    plt.figure(figsize=(5, 4))

    ax = plt.gca()


    plt.setp(ax.spines.values(), linewidth=3)

    # The ticks
    ax.xaxis.set_tick_params(width=3)
    ax.yaxis.set_tick_params(width=3)

    mol_num=data_diss_all_label['episode_i'].max()
    episode_list=[]    
    for i in range(mol_num):
        index_max_diss=data_diss_all_label[data_diss_all_label['episode_i']==i]['diss_i'].index.max()
        episode_list.append(index_max_diss)
    label_episode=data_diss_all_label.loc[episode_list]
    label_episode=label_episode.reset_index(drop=True)

    label_middle=data_diss_all_label[~(data_diss_all_label.index.isin(episode_list))]
    label_middle=label_middle.reset_index(drop=True)
    
    succ_nums=[]
    original_nums=[]
    messy_nums=[]
    interval=10
    for i in range(episode_start, episode_start+len(label_episode), interval):
        succ_num=len(label_episode[(label_episode['done_diss']=='True') &  (label_episode['episode_i']>=i) &  (label_episode['episode_i']<i+interval)])
    
        messy_num=len(label_episode[(label_episode['done_diss']=='False') &  (label_episode['episode_i']>=i) &  (label_episode['episode_i']<i+interval)])
        succ_nums.append(succ_num)

        messy_nums.append(messy_num)


    plt.plot([i for i in range(0, len(label_episode), interval)], succ_nums, color='red', linewidth=3)

    plt.plot([i for i in range(0, len(label_episode), interval)], messy_nums, color='purple', linewidth=3)




    # plt.bar([i for i in range(0, len(label_episode), interval)], succ_nums, color='red', width = 4, alpha=0.5)

    # plt.bar([i for i in range(0, len(label_episode), interval)], messy_nums, color='purple', width = 4, alpha=0.7)
    # plt.bar([i for i in range(0, len(label_episode), interval)], original_nums, color='grey', width = 4, alpha=0.8)
    plt.legend(['succ', 'messy'], frameon=False, fontsize=14, loc='upper right')

    plt.scatter([i for i in range(0, len(label_episode), interval)], succ_nums, color='red', linewidth=3)

    plt.scatter([i for i in range(0, len(label_episode), interval)], messy_nums, color='purple', linewidth=3)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(-1, 11.9)
    plt.xlim(-10, 190)
    plt.xlabel('Episodes at the interval of 10', fontsize=16)
    plt.ylabel('Numbers', fontsize=16)
    plt.title('%s' % current_time, fontsize=16)
    return label_episode, label_middle


def draw_diss_examp(episode_i, diss_i, current_time='12_5_15_15', succ=0, plot_current=False, fig_len=20, column_num=4):


    file_path='c:\\Users\\wun2\\github\\clean_reaction_rl_1209\\diss_all_output\\save_diss_image_all_sac_%s' % current_time

    diss_before_img=os.path.join(file_path, 'diss_before_img')
    diss_after_img=os.path.join(file_path, 'diss_after_img')
    diss_before_data=os.path.join(file_path, 'diss_before_data')
    diss_after_data=os.path.join(file_path, 'diss_after_data')
    diss_data=os.path.join(file_path, 'diss_data')
    diss_tip_pos=os.path.join(file_path, 'diss_tip_pos')

    plt.subplot(fig_len,column_num,(succ)*column_num+1)
    img=cv2.imread('%s/diss_before_%d_%d.png' % (diss_before_img, episode_i, diss_i), cv2.IMREAD_GRAYSCALE)
    with open('%s/scan_data_%d_%d.pkl' % (diss_before_data, episode_i, diss_i), 'rb') as f:
        scan_data_before = pickle.load(f)
    img_prop_small=mol_property(img, offset_x_nm=scan_data_before.offset_nm[0], offset_y_nm=scan_data_before.offset_nm[1], len_nm=scan_data_before.len_nm[0])
    img_prop_small.center_points_from_contour(plot_graph=False)
    img_prop_small.contour_property()
    with open('%s/vert_tip_pos_%d_%d.pkl' % (diss_tip_pos, episode_i, diss_i), 'rb') as f:
            vert_tip_pos = pickle.load(f)
    plt.scatter([vert_tip_pos[3]], [vert_tip_pos[4]], color='r')
    plt.imshow(img, extent=[scan_data_before.offset_nm[0]-scan_data_before.len_nm[0]/2,  scan_data_before.offset_nm[0]+scan_data_before.len_nm[0]/2, scan_data_before.offset_nm[1]+scan_data_before.len_nm[1], scan_data_before.offset_nm[1]])
    plt.title('%.2f' % img_prop_small.area)
    

    plt.subplot(fig_len,column_num,succ*column_num+2)
    if os.path.exists('%s/vert_data_%d_%d.pkl' % (diss_data, episode_i, diss_i)):
        with open('%s/vert_data_%d_%d.pkl' % (diss_data, episode_i, diss_i), 'rb') as f:
            vert_data = pickle.load(f)
        with open('%s/vert_tip_pos_%d_%d.pkl' % (diss_tip_pos, episode_i, diss_i), 'rb') as f:
            vert_tip_pos = pickle.load(f)
        plt.plot(vert_data.V, vert_data.topography, c='r')
        plt.scatter(vert_data.V[0], vert_data.topography[0], s=100, color='yellow')
        # if vert_tip_pos[7]:
        #     plt.plot(vert_data.V, vert_data.topography, c='r')
        # else:
        #     plt.plot(vert_data.V, vert_data.topography)

        topography=np.array(vert_data.topography)
        diff_topography=np.abs(topography[0:512].sum()-topography[512:].sum())
        # plt.title('%d_%d_%.2f_%.2f_%.2f_%s' % (episode_i, diss_i, diff_topography, vert_tip_pos[5], vert_tip_pos[6], vert_tip_pos[7]))
        plt.title('%d_%d_%.2f_%.2f' % (episode_i, diss_i, vert_tip_pos[5], vert_tip_pos[6]))
        plt.xlim(0, 3.8)
        plt.xticks([i for i in range(0,4)])
        plt.xlabel('Bias (V)')
        plt.ylabel('Topography (nm)')

  
    plt.subplot(fig_len,column_num,succ*column_num+3)
    if plot_current:
        if os.path.exists('%s/vert_data_%d_%d.pkl' % (diss_data, episode_i, diss_i)):
            with open('%s/vert_data_%d_%d.pkl' % (diss_data, episode_i, diss_i), 'rb') as f:
                vert_data = pickle.load(f)
            with open('%s/vert_tip_pos_%d_%d.pkl' % (diss_tip_pos, episode_i, diss_i), 'rb') as f:
                vert_tip_pos = pickle.load(f)
            plt.plot(vert_data.time, vert_data.current, c='r')
            # if vert_tip_pos[7]:
            #     plt.plot(vert_data.time, vert_data.current, c='r')
            # else:
            #     plt.plot(vert_data.time, vert_data.current)
            plt.xlabel('Time (s)')
            plt.ylabel('Current (A)')
            plt.subplot(fig_len,column_num,(succ)*column_num+4)
    else:
        pass
    img=cv2.imread('%s/diss_after_%d_%d.png' % (diss_after_img, episode_i, diss_i), cv2.IMREAD_GRAYSCALE)
    with open('%s/scan_data_%d_%d.pkl' % (diss_after_data, episode_i, diss_i), 'rb') as f:
        scan_data_after = pickle.load(f)
    img_prop_small=mol_property(img, offset_x_nm=scan_data_after.offset_nm[0], offset_y_nm=scan_data_after.offset_nm[1], len_nm=scan_data_after.len_nm[0])
    img_prop_small.center_points_from_contour(plot_graph=False)
    img_prop_small.contour_property()
    plt.title('%.2f' % img_prop_small.area)
    plt.scatter([vert_tip_pos[3]], [vert_tip_pos[4]], color='r')
    plt.imshow(img, extent=[scan_data_after.offset_nm[0]-scan_data_after.len_nm[0]/2,  scan_data_after.offset_nm[0]+scan_data_after.len_nm[0]/2, scan_data_after.offset_nm[1]+scan_data_after.len_nm[1], scan_data_after.offset_nm[1]])



def draw_small_mol_examp(episode_i, diss_i, current_time='12_5_15_15', degree_label=False, degree=0):


    file_path='c:\\Users\\wun2\\github\\clean_reaction_rl_1209\\diss_all_output\\save_diss_image_all_sac_%s' % current_time

    diss_before_img=os.path.join(file_path, 'diss_before_img')
    diss_after_img=os.path.join(file_path, 'diss_after_img')
    diss_before_data=os.path.join(file_path, 'diss_before_data')
    diss_after_data=os.path.join(file_path, 'diss_after_data')
    diss_data=os.path.join(file_path, 'diss_data')
    diss_tip_pos=os.path.join(file_path, 'diss_tip_pos')
    plt.figure(figsize=(4,  4))
    plt.imshow(cv2.imread('%s/diss_before_%d_%d.png' % (diss_before_img, episode_i, diss_i), cv2.IMREAD_GRAYSCALE), extent=[0,3.5,3.5,0])

    # plt.xticks([])
    # plt.yticks([])
    plt.axis('off')
    plt.hlines(3.25, 2.5, 3.3, linewidth=3, color='w')
    plt.text(2.56, 3.1, '1 nm', fontsize=16, color='w')         # default color='k'
    if degree_label:
        plt.text(0.3, 3.15, '%s°' % degree, fontsize=16, color='w')