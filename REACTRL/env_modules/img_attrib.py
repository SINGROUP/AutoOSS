import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Ellipse
from scipy.spatial.distance import cdist







class mol_property:

    def __init__(self, img, pixel=None, offset_x_nm=0, offset_y_nm=0, len_nm=None) -> None:
        if pixel is None:
            pixel = img.shape[0]
        if len_nm is None:
            len_nm = img.shape[0]

        self.pixel = pixel
        self.offset_x_nm = offset_x_nm
        self.offset_y_nm = offset_y_nm
        self.len_nm = len_nm
        self.unit_nm = self.len_nm/self.pixel
        self.img = img
    
    def detect_contours(self, method='otsu', thres_global=[50, 255], thres_otsu_blur_kernel=5):
        if method == 'otsu':
            self.ret,self.thresh = cv2.threshold(self.img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        elif method == 'global':
            self.ret,self.thresh = cv2.threshold(self.img, thres_global[0], thres_global[1], cv2.THRESH_BINARY)
        elif method == 'otsu_gaussian':
            blur = cv2.GaussianBlur(self.img, (thres_otsu_blur_kernel, thres_otsu_blur_kernel), 0)
            self.ret,self.thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.contours, _ = cv2.findContours(self.thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.contours_points_num = [len(cnt) for cnt in self.contours]
        self.contours_max = self.contours[np.argmax(self.contours_points_num)]


    def contour_property(self, cnt=None, round_digit=2):
        
        if cnt is None:
            self.detect_contours()
            cnt = self.contours_max
        self.cnt = cnt
        self.area = cv2.contourArea(cnt)*self.unit_nm*self.unit_nm
        self.perimeter = cv2.arcLength(cnt, True)*self.unit_nm
        self.convexity = cv2.isContourConvex(cnt)
        # approximate as a rectangule
        self.rect_x, self.rect_y, self.rect_w, self.rect_h = cv2.boundingRect(cnt)
        self.rect_x, self.rect_y, self.rect_w, self.rect_h = round(self.rect_x*self.unit_nm+self.offset_x_nm-self.len_nm/2, round_digit), round(self.rect_y*self.unit_nm+self.offset_y_nm, round_digit), round(self.rect_w*self.unit_nm, round_digit), round(self.rect_h*self.unit_nm, round_digit)

        # approximate as a minimum area rectangle
        self.rect_min = cv2.minAreaRect(cnt)
        self.rect_min_x, self.rect_min_y = round(self.rect_min[0][0]*self.unit_nm+self.offset_x_nm-self.len_nm/2, round_digit), round(self.rect_min[0][1]*self.unit_nm+self.offset_y_nm, round_digit)
        self.rect_min_w, self.rect_min_h = round(self.rect_min[1][0]*self.unit_nm, round_digit), round(self.rect_min[1][1]*self.unit_nm, round_digit)
        self.rect_min_angle = round(self.rect_min[2], round_digit)

        # approximate as a circle
        self.circle = cv2.minEnclosingCircle(cnt)
        (self.circle_x, self.circle_y), self.circle_radius = self.circle
        self.circle_x, self.circle_y, self.circle_radius = round(self.circle_x*self.unit_nm+self.offset_x_nm-self.len_nm/2, round_digit), round(self.circle_y*self.unit_nm+self.offset_y_nm, round_digit), round(self.circle_radius*self.unit_nm, round_digit)

        # approximate as a ellipse
        self.ellipse = cv2.fitEllipse(cnt) # (center_x, center_y), (width, height), angle
        self.ellipse_x, self.ellipse_y = round(self.ellipse[0][0]*self.unit_nm+self.offset_x_nm-self.len_nm/2, round_digit), round(self.ellipse[0][1]*self.unit_nm+self.offset_y_nm, round_digit)
        self.ellipse_width, self.ellipse_height = round(self.ellipse[1][0]*self.unit_nm, round_digit), round(self.ellipse[1][1]*self.unit_nm, round_digit)
        self.ellipse_angle = round(self.ellipse[2], round_digit)

    def center_points_from_contour(self, dist_thres=1.8, mol_area_limit=[1.5, 2.8], round_digit=2, plot_graph=True):
        selected_points_from_contours=[]
        detect_mols_from_contours=[]
        detect_mols_center_from_contours=[]
        self.detect_contours()
        for i in range(len(self.contours)):
            if self.contours_points_num[i]>10:
                cnt = self.contours[i]
                ellipse = cv2.fitEllipse(cnt) # (center_x, center_y), (width, height), angle
                ellipse_x, ellipse_y = round(ellipse[0][0]*self.unit_nm+self.offset_x_nm-self.len_nm/2, round_digit), round(ellipse[0][1]*self.unit_nm+self.offset_y_nm, round_digit)
                ellipse_width, ellipse_height = round(ellipse[1][0]*self.unit_nm, round_digit), round(ellipse[1][1]*self.unit_nm, round_digit)
                ellipse_angle = round(ellipse[2], round_digit)
                selected_points_from_contours.append([ellipse_x, ellipse_y])
                area = cv2.contourArea(cnt)*self.unit_nm*self.unit_nm
                if plot_graph:
                    plt.scatter(ellipse_x, ellipse_y, s=10, c='b')
                if area>mol_area_limit[0] and area<mol_area_limit[1]:
                    detect_mols_from_contours.append(ellipse)
                    detect_mols_center_from_contours.append([ellipse_x, ellipse_y])
                    # if plot_graph:
                    #     cv2.ellipse(self.img, ellipse, (255, 0, 0), 2)
                    #     plt.gca().add_patch(Ellipse((ellipse_x, ellipse_y), width=ellipse_width, height=ellipse_height, angle=ellipse_angle, color='yellow', fill=False))

        if plot_graph:
            plt.imshow(self.img, extent=[self.offset_x_nm-self.len_nm/2, self.offset_x_nm+self.len_nm/2, self.offset_y_nm+self.len_nm, self.offset_y_nm])
            plt.ylim(self.offset_y_nm+self.len_nm, self.offset_y_nm)
            plt.xlim(self.offset_x_nm-self.len_nm/2, self.offset_x_nm+self.len_nm/2)
        self.selected_points_from_contours = selected_points_from_contours
        self.detect_mols_from_contours = detect_mols_from_contours
        self.detect_mols_center_from_contours = detect_mols_center_from_contours




    

    def plot_contour(self, cnt=None, color=(255, 0, 0), thickness=1, ellipse=True, rect=False, circle=False, text=False):
        self.contour_property(cnt=cnt)
        cv2.drawContours(self.img, self.cnt, 0, color, thickness)
        cv2.ellipse(self.img, self.ellipse, color, thickness)
        plt.imshow(self.img, extent=[self.offset_x_nm-self.len_nm/2, self.offset_x_nm+self.len_nm/2, self.offset_y_nm+self.len_nm, self.offset_y_nm])
        plt.scatter(self.ellipse_x, self.ellipse_y, c='blue', s=20)
        if text:
            plt.text(self.ellipse_x, self.ellipse_y, 'w: %.2f h: %.2f ang: %.2f area: %.2f' % (self.ellipse_width, self.ellipse_height, self.ellipse_angle, self.area), color='red', fontsize=10)

    

    def detect_edges(self, light_limit=[100, 200], blur_kernel=5) -> tuple:

        # Display original image


        # Blur the image for better edge detection
        img_blur= cv2.GaussianBlur(self.img, (blur_kernel,blur_kernel), 0) 
        
        # Sobel Edge Detection
        sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
        sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
        sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
        # Display Sobel Edge Detection Images
        
        # Canny Edge Detection
        self.edges = cv2.Canny(image=img_blur, threshold1=light_limit[0], threshold2=light_limit[1]) # Canny Edge Detection

    
    
    def select_points(self, shape_type='blob', dist_thres=1.8, light_limit=[50, 150], s=10, plot_graph=True):
        '''selct points from detected edges for large image (such as 250 nm) and select points from dectected blobs for small image (such as 10 nm)
        shape_type: 'blob' or 'edge'
        dist_thres: the distance threshold between two points (approximate to molecule size)
        light_limit: the light limit for selecting blob points
        plot_graph: plot the selected points on the image
        '''
        
        if shape_type=='edge':
            self.detect_edges()
            detect_mols=np.where(self.edges>0)
        else:
            detect_mols=np.where(self.img>light_limit[1])

        
        data_mols ={'x': detect_mols[1], 'y': detect_mols[0]}
        data_mols=pd.DataFrame(data_mols)
        data_mols=data_mols.drop_duplicates(ignore_index=True)

        selected_points=[]
        selected_points.append([data_mols.x[0], data_mols.y[0]])
        if plot_graph:
            plt.imshow(self.img, extent=[self.offset_x_nm-self.len_nm/2, self.offset_x_nm+self.len_nm/2, self.offset_y_nm+self.len_nm, self.offset_y_nm])
            plt.scatter(self.offset_x_nm-self.len_nm/2+self.unit_nm*data_mols.x[0], self.offset_y_nm+self.unit_nm*data_mols.y[0])
        for i in range(len(data_mols)):
            selected_points_array=np.array(selected_points)
            if np.sqrt((data_mols.x[i]-selected_points_array[:, 0])**2+(data_mols.y[i]-selected_points_array[:, 1])**2).min()*self.unit_nm>dist_thres:
                selected_points.append([data_mols.x[i], data_mols.y[i]])
                if plot_graph:
                    plt.scatter(self.offset_x_nm-self.len_nm/2+self.unit_nm*data_mols.x[i], self.offset_y_nm+self.unit_nm*data_mols.y[i], s=s)

        if plot_graph:
            plt.xlim(self.offset_x_nm-self.len_nm/2, self.offset_x_nm+self.len_nm/2)
            plt.ylim(self.offset_y_nm+self.len_nm, self.offset_y_nm)


        self.selected_points_nm=[[selected_points[i][0]*self.unit_nm+self.offset_x_nm-self.len_nm/2, selected_points[i][1]*self.unit_nm+self.offset_y_nm] for i in range(len(selected_points))]
        return self.selected_points_nm
    
    def detect_mol_from_points(self, mol_dist_thres=2.5):
        '''select indivisal molecules from selected points (absolute position in points))'''
        self.select_points(plot_graph=False)
        mol_points=[]
        dist=cdist(np.array(self.selected_points_nm), np.array(self.selected_points_nm))
        for i in range(len(self.selected_points_nm)):
            if len(np.nonzero(dist[i])[0])>0:
                mol_min_dist = np.min(dist[i][np.nonzero(dist[i])])
                if mol_min_dist>mol_dist_thres:
                    mol_points.append(self.selected_points_nm[i])
        return mol_points




