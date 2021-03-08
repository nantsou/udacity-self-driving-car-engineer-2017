# -*- coding: utf-8 -*-
# import required modules
import numpy as np
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

# initialize the objects of LaneFinder
calibrator = Calibrator(calibration_path)
ptransformer = PTransformer(src=src, dst=dst)
masker = Masker()
lanefinder = LaneFinder(calibrator=calibrator, ptransformer=ptransformer, 
                        masker=masker, n_image=1, scan_image_steps=10, margin=50)

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

def combined_processor(image, test=False):
    # copy the image used for the final output
    output_image = np.copy(image)
    # get lane finding results
    warp_image, curvature, offset = lanefinder.process(image, overlay_info_only=True)
    # get vehicle detecting results
    bbox_list = vd.process(image, bbox_only=True)

    # draw overlay and put curvature and offset info
    output_image = lanefinder._draw_overlay(warp_image, output_image)
    output_image = lanefinder._put_text(output_image, curvature, offset)
    # draw box of detected vehicles
    output_image = utils.draw_bbox(output_image, bbox_list, color=(0, 255, 0))

    # reset frame_cnt and heatmap as the test inputs are discrete images
    if test:
        vd.frame_cnt = 0
        vd.hm.avg_heat = None
        vd.hm.heats = []

    return output_image

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
        #win_set = vd.get_win_set(image, vd.win_set_params)
        # get win_images
        #win_images = vd.get_win_images(image, win_set)
        # perfrom vehicle detection
        #win_out, win_prob = vd.detect(win_set, win_images)
        # visualize the primary detecting results
        #out_image = utils.draw_bbox_with_prob(image, win_out, win_prob)

        out_image = combined_processor(image, test=True)
        # save output image
        fname = 'final_with_laneline_' + image_path.split('/')[1][:-4]
        save_single_image(out_image, fname)
        # reset frame_cnt and heatmap as the test inputs are discrete images
        #vehicledetector.frame_cnt = 0
        #vehicledetector.hm.avg_heat = None
        #vehicledetector.hm.heats = []

    plt.show()