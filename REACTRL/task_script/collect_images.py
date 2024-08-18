import os
import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



def save_scan_image(
    env,
    x_nm: float = 0,
    y_nm: float = 0,
    pixel: int = 128,
    scan_mvoltage: float = 1000.0,
    scan_speed: float = 500,
    scan_len_nm: float = 10,
    save_img_folder: str = None,
    save_data_folder: str = None,
    img_name: str = None,
    data_name: str = None,
    save_img: bool = True,
    save_data: bool = True
) -> tuple:
    """scanning a image and save

    Args:
        env (_type_): createc environment.
        x_nm (_type_, optional): the offset_x_nm of scanning region.
        y_nm (_type_, optional): the offset_y_nm of scanning region.
        scan_len_nm (_type_, optional): the length of scanning region.
        save_img_folder (_type_, optional): the folder for saving images.
        filename (_type_, optional): the filename for saving images.

    Returns:
        _type_: _description_
    """        
    env.createc_controller.offset_nm = np.array([x_nm, y_nm])
    env.createc_controller.im_size_nm= scan_len_nm
    # env.createc_controller.stm.setparam('Num.X', pixel)
    # env.createc_controller.stm.setparam('Num.Y', pixel)
    env.createc_controller.stm.setparam('BiasVolt.[mV]', scan_mvoltage)
    # env.createc_controller.set_speed(scan_speed)




    #TODO add line to set pixel
    scan_data=env.createc_controller.scan_image(speed=scan_speed)
    if save_img:
        plt.imsave('%s/img_forward_%s.png' % (save_img_folder, img_name), scan_data[0])
        plt.imsave('%s/img_backward_%s.png' % (save_img_folder, img_name), scan_data[1])
        plt.imsave('%s/img_for_back_%s.png' % (save_img_folder, img_name), (scan_data[0]+scan_data[1])/2)
    if save_data:
        with open(os.path.join(save_data_folder, 'scan_data_%s.pkl' % (data_name)), "wb") as fp:
            pickle.dump(scan_data, fp)
    return scan_data



def tip_form_region(
    env,
    tip_form_ref_x: float = 280,
    tip_form_ref_y: float = -160,
    tip_form_z_range: list = [20, 35],
    tip_form_len_nm: float = 150,
    tip_form_dist_thres: float = 1,
    scan_default_z: float = 20,
    tip_form_check_points: list = None
) -> list:

    """forming tip at a specific region
    Usage example: tip_form_region(tip_form_ref_x=280, tip_form_ref_y=-160, tip_form_z_range=[20, 35], tip_form_len_nm=150, tip_form_dist_thres=1, scan_default_z=20, tip_form_check_points=None)
    Args:
        tip_form_ref_x (float, optional): the offset_x_nm of region for forming tips. 
        tip_form_ref_y (float, optional): the offset_y_nm of region for forming tips. 
        tip_form_z_range (list, optional): the range of z_nm for forming tips. 
        tip_form_len_nm (float, optional): the length of region for forming tips.
        tip_form_dist_thres (float, optional): the distance threshold for select a tip form position which should be away from previous tip forming points.
        scan_default_z (float, optional): the default z_nm for scanning after tip forming.
        tip_form_check_points (list, optional): the list of previous tip forming points. 

    
    Returns:
        _type_: _description_
    """    
    print('start tip forming now')
    tip_form_x=tip_form_ref_x-tip_form_len_nm/2+tip_form_len_nm*np.random.rand()
    tip_form_y=tip_form_ref_y+tip_form_len_nm*np.random.rand()
    if tip_form_check_points is None:
        tip_form_check_points=[]
        tip_form_check_points.append([tip_form_ref_x, tip_form_ref_y])
    tip_form_check_points_array=np.array(tip_form_check_points)
    while np.sqrt((tip_form_check_points_array[:, 0]-tip_form_x)**2+(tip_form_check_points_array[:, 1]-tip_form_y)**2).min()<tip_form_dist_thres:
        tip_form_x=tip_form_ref_x-tip_form_len_nm/2+tip_form_len_nm*np.random.rand()
        tip_form_y=tip_form_ref_y+tip_form_len_nm*np.random.rand()
    upper_limit=tip_form_z_range[1]
    lower_limit=tip_form_z_range[0]
    tip_form_z=lower_limit+np.random.rand()*(upper_limit-lower_limit)
    env.createc_controller.tip_form(tip_form_z, tip_form_x, tip_form_y, np.array([tip_form_ref_x, tip_form_ref_y]), tip_form_len_nm)  
    tip_form_check_points.append([tip_form_x, tip_form_y])
    env.createc_controller.set_Z_approach(scan_default_z)
    return tip_form_check_points




def scan_all_approach_area(
    env, # createc environment
    approach_limit: list = [-300, 300, -300, 300],
    scan_len_nm: float = 200,
    pixel: int = 128,
    scan_speed: float = 1500,
    total_output_folder: str = 'all_output',
    task_folder_prefix: str = 'approach',
    save_scan_img_prefix: str = 'scan_img',
    save_scan_data_prefix: str = 'scan_data',
    plot_scan_points: bool = True,
    tip_form_ref_x: float = 280,
    tip_form_ref_y: float = -160,
    tip_form_z_range: list = [20, 35],
    tip_form_len_nm: float = 150,
    tip_form_dist_thres: float = 1,
    scan_default_z: float = 20,
    tip_form_check_points: list = None
) -> None:

    """scan all approach area with a small scanning image size and form tip at a specific region
    Usage example: scan_all_approach_area(approach_limit=[-320, 320, -320, 320], scan_len_nm=200, tip_form_ref_x=280, tip_form_ref_y=-160, tip_form_z_range=[20, 35], tip_form_len_nm=150, tip_form_dist_thres=1, scan_default_z=20, tip_form_check_points=None):

    Args:
        approach_limit (list, optional): the left, right, bottom, top limit of approach area. 
        scan_len_nm (float, optional): the length of scan area for each time.
        total_output_folder (str, optional): the folder for saving all output.
        task_folder_prefix (str, optional): the prefix of folder for saving current task output.
        save_scan_img_prefix (str, optional): the prefix of folder for saving scan images.
        save_scan_data_prefix (str, optional): the prefix of folder for saving scan data.
        plot_scan_points (bool, optional): whether plot the scan points on the approach area.
        other parameters in function tip_form_region
    """    

    current_time=datetime.datetime.now()    
    task_folder='%s/%s_%s_%s_%s_%s' % (total_output_folder, task_folder_prefix, current_time.month, current_time.day, current_time.hour, current_time.minute)
    save_scan_img='%s/%s_%s_%s_%s_%s' % (task_folder, save_scan_img_prefix, current_time.month, current_time.day, current_time.hour, current_time.minute)
    save_scan_data='%s/%s_%s_%s_%s_%s' % (task_folder, save_scan_data_prefix, current_time.month, current_time.day, current_time.hour, current_time.minute)

    # check if these folders exist
    if not os.path.exists(total_output_folder):
        os.mkdir(total_output_folder)
    if not os.path.exists(task_folder):
        os.mkdir(task_folder)
    if not os.path.exists(save_scan_img):
        os.mkdir(save_scan_img)
    if not os.path.exists(save_scan_data):
        os.mkdir(save_scan_data)

    # seperate the approach area into small scanning regions
    num_x=(approach_limit[1]-approach_limit[0])/scan_len_nm
    num_y=(approach_limit[3]-approach_limit[2])/scan_len_nm
    num_xy=int(np.min([num_x, num_y]))

    for i in tqdm(range(num_xy)):
        for j in range(num_xy):
            x_nm=approach_limit[0]+scan_len_nm/2+i*scan_len_nm
            y_nm=approach_limit[2]+j*scan_len_nm
            print('scanning regions', i, j, x_nm, y_nm, scan_len_nm)
            if plot_scan_points:
                plt.scatter(x_nm, y_nm, c='r')
            else:
                scan_data=save_scan_image(env, x_nm=x_nm, y_nm=y_nm, scan_speed=scan_speed, scan_len_nm=scan_len_nm, pixel=pixel, save_img_folder=save_scan_img, save_data_folder=save_scan_data, img_name='%s_%s' % (i, j), data_name='%s_%s' % (i, j))

                # if j%5==4:
                #     print('start tip forming now')
                #     tip_form_check_points=tip_form_region(env, tip_form_ref_x=tip_form_ref_x, tip_form_ref_y=tip_form_ref_y, tip_form_z_range=tip_form_z_range, tip_form_len_nm=tip_form_len_nm, tip_form_dist_thres=tip_form_dist_thres, scan_default_z=scan_default_z, tip_form_check_points=tip_form_check_points)
    
    save_scan_image(env, x_nm=x_nm, y_nm=y_nm, scan_speed=scan_speed, scan_len_nm=scan_len_nm, save_img=False, save_data=False)
    if plot_scan_points:
        plt.xlim(approach_limit[0], approach_limit[1])
        plt.ylim(approach_limit[2], approach_limit[3])




        