# -*- coding: utf-8 -*-
# import required modules
import os
import cv2
import numpy as np
import pickle
# import LaneFinder module
from LaneFinder.calibrator import Calibrator
from LaneFinder.ptransformer import PTransformer
from LaneFinder.masker import Masker
from LaneFinder.lanefinder import LaneFinder
from LaneFinder.line import Line
# import VehicleDector module
from VehicleDetector import utils
from VehicleDetector.featureextractor import FeatureExtractor
from VehicleDetector.vehicledector import VehicleDetector
from VehicleDetector.heatmap import HeatMap

# define the windows for the perspective transform
src = np.float32([
    [595, 450],
    [690, 450],
    [1115, 720],
    [216, 720]
])

dst = np.float32([
    [450, 0],
    [830, 0],
    [830, 720],
    [450, 720]
])

# define the path of training outputs
svc_path = 'outputs/svc.p'
x_scaler_path = 'outputs/x_scaler.p'
calibration_path = 'outputs/calibration.p'

# load SVM model
clf = utils.load_content(svc_path)
# load scaler
X_scaler = utils.load_content(x_scaler_path)

# define the param of sliding windows
width = 1280
height = 720

params = []
params.append(utils.set_win_set_param([None, None], [height//2, np.int(height*0.7)], (80, 80), (0.8, 0.8)))
params.append(utils.set_win_set_param([None, None], [height//2 + 20, np.int(height*0.75)], (100, 100), (0.8, 0.8)))
params.append(utils.set_win_set_param([None, None], [height//2 + 50, np.int(height*0.80)], (120, 120), (0.8, 0.8)))
params.append(utils.set_win_set_param([None, None], [height//2 + 80, np.int(height*0.92)], (150, 150), (0.8, 0.8)))

# initialize the objects of LaneFinder
calibrator = Calibrator(calibration_path)
ptransformer = PTransformer(src=src, dst=dst)
masker = Masker()
lanefinder = LaneFinder(calibrator=calibrator, ptransformer=ptransformer, 
                        masker=masker, n_image=5, scan_image_steps=10, margin=50)

# initialize the objects of VehicleDetector
fe = FeatureExtractor(orient=9, ppc=8, cpb=2, cspace="YUV", ssize=(32, 32))
vehicledetector = VehicleDetector(clf=clf, x_scaler=X_scaler, fe=fe, n_frame=3, heatmap_threshold=2, win_set_params=params)

def combined_processor(image):
    # copy the image used for the final output
    output_image = np.copy(image)
    # get lane finding results
    warp_image, curvature, offset = lanefinder.process(image, overlay_info_only=True)
    # get vehicle detecting results
    bbox_list = vehicledetector.process(image, bbox_only=True)

    # draw overlay and put curvature and offset info
    output_image = lanefinder._draw_overlay(warp_image, output_image)
    output_image = lanefinder._put_text(output_image, curvature, offset)
    # draw box of detected vehicles
    output_image = utils.draw_bbox(output_image, bbox_list, color=(0, 255, 0))

    return output_image
def save_single_image(image, fname, cmap=None):
    fig = plt.figure(figsize=(8, 4.5))
    plt.imshow(image, cmap=cmap)
    a=fig.gca()
    a.set_frame_on(False)
    a.set_xticks([]); a.set_yticks([])
    plt.axis('off')
    plt.savefig('output_images/viz_dv/dv_{}.png'.format(fname), 
                bbox_inches='tight',
                transparent="True", 
                pad_inches=0)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    image = cv2.imread('test_images/test6.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # copy the image used for output image
    input_image = np.copy(image)
    save_single_image(input_image, 'input_image')
    win_set = []
    # full scan sliding window
    for idx, param in enumerate(params):
        tmp_img = np.copy(image)
        tmp_win_set = utils.slide_window(tmp_img,
                                         x_start_stop=param['x_start_stop'], 
                                         y_start_stop=param['y_start_stop'], 
                                         xy_window=param['xy_window'], 
                                         xy_overlap=param['xy_overlap'])
        tmp_win_set_img = utils.draw_bbox(tmp_img, tmp_win_set, thick=2)
        win_set += tmp_win_set
        save_single_image(tmp_win_set_img, 'fs_win_set_{}'.format(idx+1))

    # detecting car
    ## get the image set of the whole windows
    win_images = vehicledetector.get_win_images(image, win_set)
    
    ## detect the vehicles
    win_out, win_prob = vehicledetector.detect(win_set, win_images)
    detect_raw_image = utils.draw_bbox_with_prob(image, win_out, win_prob)
    save_single_image(detect_raw_image, 'win_detecting_result')

    ## get the bbox after heatmap processing
    ### add heatmap
    vehicledetector.hm.add_heat(image, win_out)
    heatmap = vehicledetector.hm.avg_heat
    heat_img = np.clip(heatmap, 0, 255)
    save_single_image(heat_img, 'heatmp', cmap='hot')
    ### apply threshold that 2 is used in VehicleDetector of this project
    vehicledetector.hm.apply_threshold(threshold=2)
    filtered_heatmap = vehicledetector.hm.avg_heat
    heat_img = np.clip(filtered_heatmap, 0, 255)
    save_single_image(heat_img, 'filtered_heatmp', cmap='hot')
    ### get labels and turned it into bbox
    labels = vehicledetector.hm.get_heatmap_labels(image, win_out)
    bbox_list = vehicledetector.label_to_bbox(labels)
    output_image = utils.draw_bbox(input_image, bbox_list, color=(0, 255, 0))
    save_single_image(output_image, 'vehicle_detecting_result')

    ## perform small region scan
    ## get the fouth points which defines the region to be searched
    x_start, x_stop, y_start, y_stop = vehicledetector.get_tracking_region(bbox_list)

    ## the following procedure is copied from method, get_tracking_win_set, in VehicleDetector
    ## define the margins of x and y for detecting the area around the detected vehicles
    x_margin = 25
    y_margin = 10

    # prevent x_start which outside the image
    x_start = np.maximum(x_start - x_margin, 0)
    x_stop = np.maximum(x_stop + x_margin, image.shape[1])
    ## prevent that y_start is higher than half height of the image
    y_start = np.maximum(y_start - y_margin, 360)
    ## prevent that y_stop is lower than the hood of car
    y_stop = np.minimum(y_stop + y_margin, np.int(720 * 0.92))

    for idx, size in enumerate([64, 80, 100, 120, 150]):
        tmp_img = np.copy(image)
        tmp_win_set = utils.slide_window(tmp_img,
                                         x_start_stop=[x_start, x_stop],
                                         y_start_stop=[y_start, y_stop],
                                         xy_window=(size, size),
                                         xy_overlap=(0.8, 0.8))
        tmp_win_set_img = utils.draw_bbox(tmp_img, tmp_win_set, thick=2)
        save_single_image(tmp_win_set_img, 'ts_win_set_{}'.format(idx+1))

    ## combine lanefinder and vehicledetector result
    final_result = combined_processor(image)
    save_single_image(final_result, 'final_result')
    plt.show()
    
