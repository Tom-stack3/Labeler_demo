from cv2 import resize
from depth_analyzer import DepthAnalyzer
from points_analyzer import PointsAnalyzer
from utils.logger import Logger

import yaml
import time
import cv2


class Labeler:
    def __init__(self) -> None:
        with open('settings.yaml', 'r') as input_file:
            self.settings = yaml.load(input_file, yaml.FullLoader)

        self.logger = Logger(self.settings["output"]["output_extension"])
        self.depth_analyzer = DepthAnalyzer(self.logger)
        self.poinys_analyzer = PointsAnalyzer(self.logger)

    def label_image(self, frame):

        print(f'Original Dimensions: {frame.shape}')

        # If needed to resize the frame
        if self.settings["output"]["resize_frame"]:
            Labeler.resize_frame(frame)
            print(f'Resized Dimensions: {frame.shape}')

        img_name = Labeler.generate_img_name()

        # Calculate and save the depths of the image
        self.depth_analyzer.calc_depth(frame, img_name)

        self.poinys_analyzer.analyze_image(frame, img_name)
        # Censor the eyes of the person in the image
        self.poinys_analyzer.censor_eyes()
        # Save the image with only censored eyes
        self.poinys_analyzer.save_img("EMPTY")

        # Detect and draw points on frame
        self.poinys_analyzer.draw_points()
        # Save the image with points drawn on
        self.poinys_analyzer.save_img("DRAWN")

        # Save points and data to log
        self.poinys_analyzer.log_img_info()

    def resize_frame(frame):
        scale_percent = 60  # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)

        # Resize image
        resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        print('Resized Dimensions : ', resized.shape)
        frame = resized

    def generate_img_name():
        return time.strftime('%Y_%m_%d %H_%M_%S', time.localtime(time.time()))
