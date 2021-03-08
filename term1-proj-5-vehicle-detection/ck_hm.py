# -*- coding: utf-8 -*-
# import required modules
import numpy as np
# import VehicleDetector module
from VehicleDetector import utils
from VehicleDetector.featureextractor import FeatureExtractor
from VehicleDetector.vehicledector import VehicleDetector
from VehicleDetector.heatmap import HeatMap

# define the path of training outputs
svc_path = 'outputs/svc.p'
x_scaler_path = 'outputs/x_scaler.p'

# load SVM model
clf = utils.load_content(svc_path)
# load scaler
X_scaler = utils.load_content(x_scaler_path)

# define the param of sliding windows
## The default size of the video is 1280*720
width = 1280
height = 720
## define the window set parameters
params = []
params.append(utils.set_win_set_param([None, None], [height//2, np.int(height*0.7)], (80, 80), (0.8, 0.8)))
params.append(utils.set_win_set_param([None, None], [height//2 + 20, np.int(height*0.75)], (100, 100), (0.8, 0.8)))
params.append(utils.set_win_set_param([None, None], [height//2 + 50, np.int(height*0.80)], (120, 120), (0.8, 0.8)))
params.append(utils.set_win_set_param([None, None], [height//2 + 80, np.int(height*0.92)], (150, 150), (0.8, 0.8)))

# initialize the objects of VehicleDetector
fe = FeatureExtractor(orient=9, ppc=8, cpb=2, cspace="YUV", ssize=(32, 32))
vd = VehicleDetector(clf=clf, x_scaler=X_scaler, fe=fe, n_frame=1, heatmap_threshold=2, win_set_params=params)

def save_single_image(image, fname, cmap=None):
    fig = plt.figure(figsize=(8, 4.5))
    plt.imshow(image, cmap=cmap)
    a=fig.gca()
    a.set_frame_on(False)
    a.set_xticks([]); a.set_yticks([])
    plt.axis('off')
    plt.savefig('output_images/ck_dv_{}.png'.format(fname), 
                bbox_inches='tight',
                transparent="True", 
                pad_inches=0)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import cv2
    from glob import glob

    image_paths = glob('test_images/test*.jpg')
    out_images = []
    for image_path in image_paths:
        # load the testing image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # get win_set
        win_set = vd.get_win_set(image, vd.win_set_params)
        # get win_images
        win_images = vd.get_win_images(image, win_set)
        # perfrom vehicle detection
        win_out, win_prob = vd.detect(win_set, win_images)
        # add heatmap
        vd.hm.add_heat(image, win_out)
        heatmap = vd.hm.avg_heat
        heat_img = np.clip(heatmap, 0, 255)
        fname = 'heatmap_raw_' + image_path.split('/')[1][:-4]
        save_single_image(heat_img, fname, cmap='hot')
        # apply threshold
        vd.hm.apply_threshold(threshold=2)
        filtered_heatmap = vd.hm.avg_heat
        heat_img = np.clip(filtered_heatmap, 0, 255)
        # save output image
        fname = 'heatmap_filtered' + image_path.split('/')[1][:-4]
        save_single_image(heat_img, fname, cmap='hot')
        # reset frame_cnt and heatmap as the test inputs are discrete images
        #vehicledetector.frame_cnt = 0
        vd.hm.avg_heat = None
        vd.hm.heats = []

    plt.show()