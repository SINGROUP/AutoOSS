import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from matplotlib import pyplot as plt
import matplotlib
import cv2
import numpy as np
import pandas as pd
from collections import deque, namedtuple
from matplotlib import pyplot as plt, patches
import torch
from REACTRL    import mol_property



matplotlib.rcParams['image.cmap'] = 'gray'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
from matplotlib import pyplot as plt, patches


from matplotlib.patches import Rectangle 

# Python code to demonstrate namedtuple()
from collections import namedtuple

import pickle
from scipy.spatial.distance import cdist
import shutil


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from matplotlib import patches

def transfer_directory(current_time=None, model='sac', episodes=130, diss_maxtime=10, succ=0, succ_diss=0):
    
    total_output_folder = 'all_output'
    task_folder_prefix = 'vertmani'
    save_scan_img_large_prefix = 'scan_img_large'
    save_scan_img_before_prefix = 'scan_img_before'
    save_scan_img_after_prefix = 'scan_img_after'
    save_scan_data_large_prefix = 'scan_data_large'
    save_scan_data_before_prefix = 'scan_data_before'
    save_scan_data_after_prefix = 'scan_data_after'
    save_vert_data_prefix = 'vert_data'
    save_vert_tip_pos_prefix = 'vert_tip_pos'

    task_folder='%s/%s_%s' % (total_output_folder, task_folder_prefix, current_time)
    save_img_large='%s/%s_%s' % (task_folder, save_scan_img_large_prefix, current_time)
    save_img_before='%s/%s_%s' % (task_folder, save_scan_img_before_prefix, current_time)
    save_img_after='%s/%s_%s' % (task_folder, save_scan_img_after_prefix, current_time)
    save_data_large='%s/%s_%s' % (task_folder, save_scan_data_large_prefix, current_time)
    save_data_before='%s/%s_%s' % (task_folder, save_scan_data_before_prefix, current_time)
    save_data_after='%s/%s_%s' % (task_folder, save_scan_data_after_prefix, current_time)
    save_vert_data='%s/%s_%s' % (task_folder, save_vert_data_prefix, current_time)
    save_vert_tip_pos='%s/%s_%s' % (task_folder, save_vert_tip_pos_prefix, current_time)



    save_all_diss_name='save_diss_image_all_'+model+'_'+current_time
    if not os.path.exists(save_all_diss_name):
        os.mkdir(save_all_diss_name)
        os.mkdir(os.path.join(save_all_diss_name, 'diss_before_img'))
        os.mkdir(os.path.join(save_all_diss_name, 'diss_after_img'))
        os.mkdir(os.path.join(save_all_diss_name, 'diss_after_data'))
        os.mkdir(os.path.join(save_all_diss_name, 'diss_before_data'))
        os.mkdir(os.path.join(save_all_diss_name, 'diss_tip_pos'))  
        os.mkdir(os.path.join(save_all_diss_name, 'diss_data'))


    with open(os.path.join(save_all_diss_name, 'all_tip_pos_data_label.txt'), 'a') as f:
        f.write('episode_i \t diss_i \t done_diss \t tip_x \t tip_y \t mvoltage \t pcurrent \n')

    for episode_i in range(episodes):
        for diss_i in range(diss_maxtime):
            if os.path.exists('%s/img_forward_%d_%d.png' % (save_img_after, episode_i, diss_i)):
                max_episode=episode_i
                succ+=1
                # plt.subplot(episodes*diss_maxtime,column_num,episode_i*diss_maxtime*column_num+diss_i*column_num+3)
            else:
                continue

            # save images before dissocation
            if diss_i!=0:
                if os.path.exists('%s/img_forward_%d_%d_adjust.png' % (save_img_after,  episode_i, diss_i-1)):
                    img2=cv2.imread('%s/img_forward_%d_%d.png' % (save_img_after,  episode_i, diss_i-1), cv2.IMREAD_GRAYSCALE)
                    shutil.copyfile('%s/scan_data_%d_%d_adjust.pkl' % (save_data_after, episode_i, diss_i-1), '%s/diss_before_data/scan_data_%d_%d.pkl' % (save_all_diss_name, episode_i, diss_i))
                elif os.path.exists('%s/img_forward_%d_%d.png' % (save_img_after,  episode_i, diss_i-1)):
                    img2=cv2.imread('%s/img_forward_%d_%d.png' % (save_img_after,  episode_i, diss_i-1), cv2.IMREAD_GRAYSCALE)
                    shutil.copyfile('%s/scan_data_%d_%d.pkl' % (save_data_after, episode_i, diss_i-1), '%s/diss_before_data/scan_data_%d_%d.pkl' % (save_all_diss_name, episode_i, diss_i))
                cv2.imwrite(os.path.join(save_all_diss_name, 'diss_before_img', 'diss_before_%d_%d.png' % (episode_i, diss_i)), img2)
                
            else:
                img2=cv2.imread('%s/img_forward_%d.png' % (save_img_before, episode_i), cv2.IMREAD_GRAYSCALE)
                cv2.imwrite(os.path.join(save_all_diss_name, 'diss_before_img', 'diss_before_%d_%d.png' % (episode_i, diss_i)), img2)
                shutil.copyfile('%s/scan_data_%d.pkl' % (save_data_before, episode_i), '%s/diss_before_data/scan_data_%d_%d.pkl' % (save_all_diss_name, episode_i, diss_i))

            # save images after dissocation
            img1=cv2.imread('%s/img_forward_%d_%d.png' % (save_img_after,  episode_i, diss_i), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(os.path.join(save_all_diss_name, 'diss_after_img', 'diss_after_%d_%d.png' % (episode_i, diss_i)), img1)
            shutil.copyfile('%s/scan_data_%d_%d.pkl' % (save_data_after, episode_i, diss_i), '%s/diss_after_data/scan_data_%d_%d.pkl' % (save_all_diss_name,  episode_i, diss_i))

            # save tip position
            shutil.copyfile('%s/vert_tip_pos_%d_%d.pkl' % (save_vert_tip_pos, episode_i, diss_i), '%s/diss_tip_pos/vert_tip_pos_%d_%d.pkl' % (save_all_diss_name, episode_i, diss_i))
            with open('%s/vert_tip_pos_%d_%d.pkl' % (save_vert_tip_pos, episode_i, diss_i), 'rb') as f_tip:
                vert_tip_pos=pickle.load(f_tip)

            with open(os.path.join(save_all_diss_name, 'all_tip_pos_data_label.txt'), 'a') as f:
                f.write('%d \t %d \t %s \t %.2f \t %.2f \t %.2f \t %.2f\n' % (episode_i, diss_i, vert_tip_pos[-2], vert_tip_pos[-1][0], vert_tip_pos[-1][1], vert_tip_pos[-1][2], vert_tip_pos[-1][3]))

            if vert_tip_pos[-2]==True:
                succ_diss+=1




            # save vert data
            shutil.copyfile('%s/vert_data_%d_%d.pkl' % (save_vert_data, episode_i, diss_i), '%s/diss_data/vert_data_%d_%d.pkl' % (save_all_diss_name, episode_i, diss_i))

    with open(os.path.join(save_all_diss_name, 'README.txt'), 'a') as readme:
        readme.write('current_time: %s, model: %s, episodes: %d, diss_maxtime: %d, sample num: %d, succ diss num: %d\n' % (current_time, model, max_episode, diss_maxtime, succ, succ_diss))


def plot_all_diss(current_time=None, model='sac', episodes=130, diss_maxtime=10, diss_maxtime_plot=400, succ=0, column_num=4):
    plt.figure(figsize=(15,650))
    all_tip_pos=[]
    all_diss_data=[]
    mol_center_x_nm_before=[]
    mol_center_y_nm_before=[]
    for episode_i in range(episodes):

        plt.subplot(diss_maxtime_plot,column_num,(succ+1)*column_num+1)
        if os.path.exists('%s/img_forward_%d.png' % (save_img_before, episode_i)):
            
            img=cv2.imread('%s/img_forward_%d.png' % (save_img_before, episode_i), cv2.IMREAD_GRAYSCALE)
            with open('%s/scan_data_%d.pkl' % (save_data_before, episode_i), 'rb') as f:
                scan_data_after = pickle.load(f)
            img_prop_small=mol_property(img, offset_x_nm=scan_data_after.offset_nm[0], offset_y_nm=scan_data_after.offset_nm[1], len_nm=scan_data_after.len_nm[0])
            img_prop_small.center_points_from_contour(plot_graph=False)
            img_prop_small.contour_property()
            mol_center_x_nm=img_prop_small.ellipse_x
            mol_center_y_nm=img_prop_small.ellipse_y
            mol_center_x_nm_before.append(mol_center_x_nm)
            mol_center_y_nm_before.append(mol_center_y_nm) 

            plt.imshow(img, extent=[scan_data_after.offset_nm[0]-scan_data_after.len_nm[0]/2,  scan_data_after.offset_nm[0]+scan_data_after.len_nm[0]/2, scan_data_after.offset_nm[1]+scan_data_after.len_nm[1], scan_data_after.offset_nm[1]])


        for diss_i in range(diss_maxtime):
            if os.path.exists('%s/img_forward_%d_%d.png' % (save_img_after, episode_i, diss_i)):
                succ+=1
            else:
                continue    

            plt.subplot(diss_maxtime_plot,column_num,succ*column_num+3)
            img=cv2.imread('%s/img_forward_%d_%d.png' % (save_img_after,  episode_i, diss_i), cv2.IMREAD_GRAYSCALE)

            with open('%s/vert_tip_pos_%d_%d.pkl' % (save_vert_tip_pos,  episode_i, diss_i), 'rb') as f:
                vet_tip_pos = pickle.load(f)
            with open('%s/scan_data_%d_%d.pkl' % (save_data_after,  episode_i, diss_i), 'rb') as f:
                scan_data_after = pickle.load(f)

            try:
                img_prop_small=mol_property(img, offset_x_nm=scan_data_after.offset_nm[0], offset_y_nm=scan_data_after.offset_nm[1], len_nm=scan_data_after.len_nm[0])
                img_prop_small.center_points_from_contour(plot_graph=False)
                img_prop_small.contour_property()
                plt.scatter(vet_tip_pos[1], vet_tip_pos[2], c='b', s=10)
                plt.gca().add_patch(patches.Circle((vet_tip_pos[1], vet_tip_pos[2]), diss_radius_nm, color='b', fill=False))
                plt.scatter(vet_tip_pos[3], vet_tip_pos[4], c='r', s=10)
                all_tip_pos.append(vet_tip_pos)

                plt.title('after_%d_%d_%.2f_%.2f' % (episode_i, diss_i, scan_data_after.offset_nm[0], img_prop_small.area))
            except:
                pass
            plt.imshow(img, extent=[scan_data_after.offset_nm[0]-scan_data_after.len_nm[0]/2,  scan_data_after.offset_nm[0]+scan_data_after.len_nm[0]/2, scan_data_after.offset_nm[1]+scan_data_after.len_nm[1], scan_data_after.offset_nm[1]])
        
        

            # display adjusted images after dissocation
            plt.subplot(diss_maxtime_plot,column_num,succ*column_num+4)
            if os.path.exists('%s/img_forward_%d_%d_adjust.png' % (save_img_after,  episode_i, diss_i)):
                
                img=cv2.imread('%s/img_forward_%d_%d_adjust.png' % (save_img_after,  episode_i, diss_i), cv2.IMREAD_GRAYSCALE)

                with open('%s/scan_data_%d_%d_adjust.pkl' % (save_data_after,  episode_i, diss_i), 'rb') as f:
                    scan_data_after = pickle.load(f)


                if os.path.exists('%s/vert_tip_pos_%d_%d.pkl' % (save_vert_tip_pos, episode_i, diss_i+1)):
                    with open('%s/vert_tip_pos_%d_%d.pkl' % (save_vert_tip_pos,  episode_i, diss_i+1), 'rb') as f:
                        vert_tip_pos = pickle.load(f)
                    plt.scatter(vert_tip_pos[3], vert_tip_pos[4], c='r', s=10)

                try:
                    img_prop_small=mol_property(img, offset_x_nm=scan_data_after.offset_nm[0], offset_y_nm=scan_data_after.offset_nm[1], len_nm=scan_data_after.len_nm[0])
                    img_prop_small.center_points_from_contour(plot_graph=False)
                    img_prop_small.contour_property() 
                    plt.title('after_adjust_%d_%d_%.2f_%.2f' % (episode_i, diss_i, scan_data_after.offset_nm[0], img_prop_small.area))
                except:
                    pass
                plt.imshow(img, extent=[scan_data_after.offset_nm[0]-scan_data_after.len_nm[0]/2,  scan_data_after.offset_nm[0]+scan_data_after.len_nm[0]/2, scan_data_after.offset_nm[1]+scan_data_after.len_nm[1], scan_data_after.offset_nm[1]])
            
            # Display dissciation curves
            plt.subplot(diss_maxtime_plot,column_num,succ*column_num+2)
            if os.path.exists('%s/vert_data_%d_%d.pkl' % (save_vert_data, episode_i, diss_i)):
                with open('%s/vert_data_%d_%d.pkl' % (save_vert_data, episode_i, diss_i), 'rb') as f:
                    vert_data = pickle.load(f)
                with open('%s/vert_tip_pos_%d_%d.pkl' % (save_vert_tip_pos, episode_i, diss_i), 'rb') as f:
                    vert_tip_pos = pickle.load(f)
                if vert_tip_pos[7]:
                    plt.plot(vert_data.V, vert_data.topography, c='r')
                else:
                    plt.plot(vert_data.V, vert_data.topography)
                all_diss_data.append(vert_data)
                topography=np.array(vert_data.topography)
                diff_topography=np.abs(topography[0:512].sum()-topography[512:].sum())
                plt.title('%d_%d_%.2f_%.2f_%.2f_%s' % (episode_i, diss_i, diff_topography, vert_tip_pos[5], vert_tip_pos[6], vert_tip_pos[7]))
                
    plt.subplots_adjust(wspace=0.5, hspace=0.9)

    plt.savefig('all_output/vertmani_%s_%s.png' % (model, current_time))

    all_tip_pos_data_diss=pd.DataFrame(all_tip_pos, columns=['label', 'mol_center_x_nm', 'mol_center_y_nm', 'diss_x_nm', 'diss_y_nm', 'diss_mvoltage', 'diss_pcurrent', 'done_diss'])
    all_tip_pos_data_diss['relative_diss_x']=all_tip_pos_data_diss['diss_x_nm']-all_tip_pos_data_diss['mol_center_x_nm']
    all_tip_pos_data_diss['relative_diss_y']=all_tip_pos_data_diss['diss_y_nm']-all_tip_pos_data_diss['mol_center_y_nm']

    all_tip_pos_data_diss['mol_x_before']=mol_center_x_nm_before
    all_tip_pos_data_diss['mol_y_before']=mol_center_y_nm_before
    all_tip_pos_data_diss['relative_diss_x_before']=all_tip_pos_data_diss['diss_x_nm']-all_tip_pos_data_diss['mol_x_before']
    all_tip_pos_data_diss['relative_diss_y_before']=all_tip_pos_data_diss['diss_y_nm']-all_tip_pos_data_diss['mol_y_before']
    # all_tip_pos_data_diss['diss_data']=all_diss_data
    all_tip_pos_data_diss.to_pickle('all_output/all_tip_pos_data_diss_%s_%s.pkl' % (model, current_time))

    return all_tip_pos_data_diss

def plot_trajectory(data, scan_len_nm_small=3.5):
    
    fig = plt.figure()
    # The spines
    fig, ax = plt.subplots()


    plt.setp(ax.spines.values(), linewidth=3)

    # The ticks
    ax.xaxis.set_tick_params(width=3)
    ax.yaxis.set_tick_params(width=3)


    for i in range(len(data)):
            if i % 8 == 0:
                plt.text(data['mol_center_x_nm'][i], data['mol_center_y_nm'][i], str(i))
            if data['done_diss'][i] == True:
                plt.scatter(data['mol_center_x_nm'][i], data['mol_center_y_nm'][i], c='salmon', edgecolor='none', s=50)
                # plt.gca().add_patch(patches.Rectangle((data['mol_center_x_nm'][i]-scan_len_nm_small/2, data['mol_center_y_nm'][i]-scan_len_nm_small/2), scan_len_nm_small, scan_len_nm_small, fill=False, linewidth=3))
                # continue
            else:
                plt.scatter(data['mol_center_x_nm'][i], data['mol_center_y_nm'][i], c='blue', edgecolor='none', s=50)
                continue

    plt.xlabel('x (nm)', fontsize=16)
    plt.ylabel('y (nm)', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(320, -320)
    # plt.xlim(-320, 320)
    plt.plot(data['mol_center_x_nm'], data['mol_center_y_nm'], 'gray',  linewidth=3, zorder=0)


def make_animation(data, movie_name='movie_test.mp4'):

# Define the meta data for the movie

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='a red circle following a blue sine wave')
    writer = FFMpegWriter(fps=15, metadata=metadata)

    # Initialize the movie
    fig = plt.figure()
    # The spines
    fig, ax = plt.subplots()


    plt.setp(ax.spines.values(), linewidth=3)

    # The ticks
    ax.xaxis.set_tick_params(width=3)
    ax.yaxis.set_tick_params(width=3)

    # plot the sine wave line
    sine_line, = plt.plot(data['mol_center_x_nm'], data['mol_center_y_nm'], 'gray',  linewidth=3, zorder=0)
    red_circle, = plt.plot([], [], 'ro', markersize = 10)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('x (nm)', fontsize=20)
    plt.ylabel('y (nm)', fontsize=20)

    plt.tight_layout()
    plt.ylim(340, -340)
    plt.xlim(-340, 50)

    # Update the frames for the movie
    with writer.saving(fig, movie_name, 100):
        for i in range(len(data)):
            if i % 8 == 0:
                plt.text(data['mol_center_x_nm'][i], data['mol_center_y_nm'][i], str(i))
            if  data['done_diss'][i] == True:
                plt.scatter(data['mol_center_x_nm'][i], data['mol_center_y_nm'][i], c='salmon', edgecolor='none', s=50)
            else:
                plt.scatter(data['mol_center_x_nm'][i], data['mol_center_y_nm'][i], c='blue', edgecolor='none', s=50)
            x0, y0 = data['mol_center_x_nm'][i], data['mol_center_y_nm'][i]
            red_circle.set_data(x0, y0)
            writer.grab_frame()


def plot_graph_formate(xlabel='Dissociation number', ylabel='Current (A)'):
    

    fig, ax = plt.subplots()
    plt.setp(ax.spines.values(), linewidth=3)

    # The labels
    plt.ylabel(xlabel, fontsize=16)
    plt.xlabel(ylabel, fontsize=16)

    # The ticks
    ax.xaxis.set_tick_params(width=3)
    ax.yaxis.set_tick_params(width=3)
    ax.legend(fontsize=12)

    

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
