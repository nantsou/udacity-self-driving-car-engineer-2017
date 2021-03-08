# -*- coding: utf-8 -*-
# import required modules
import numpy as np
# import LaneFinder module
from LaneFinder.calibrator import Calibrator
from LaneFinder.ptransformer import PTransformer
from LaneFinder.masker import Masker
from LaneFinder.lanefinder import LaneFinder
# import VehicleDetector module
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
## The default size of the video is 1280*720
width = 1280
height = 720
## define the window set parameters
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


if __name__ == '__main__':
    import sys
    from moviepy.editor import VideoFileClip
    video_path = sys.argv[1] if len(sys.argv) > 1 else 'videos/project_video.mp4'
    clip1 = VideoFileClip(video_path)
    project_clip = clip1.fl_image(combined_processor)
    project_output = video_path[:-4] + '_result.mp4'
    project_clip.write_videofile(project_output, audio=False)