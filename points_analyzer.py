import yaml
import cv2
import utils.math
import math
import numpy as np


class PointsAnalyzer:
    def __init__(self, logger) -> None:
        # Load model settings from yaml
        with open('settings.yaml', 'r') as input_file:
            self.settings = yaml.load(input_file)["2D_model"]

        # Save the logger
        self.logger = logger

        # Load the Model
        protoFile = self.settings["proto_file"]
        weightsFile = self.settings["weights_file"]
        self.net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
        self.net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)


    def censor_eyes(frame, left_eye, right_eye):
        """
        censor the eyes of the person in the image
        :param frame: frame to draw on
        :param left_eye: left eye (x,y)
        :param right_eye: right eye (x,y)
        :return: None
        """

        if left_eye is None or right_eye is None:
            print("Error! Couldn't find eyes in the image, so couldn't censor them")
            return

        # calculate the slope intercept form of the line from the Left Eye to the Right eye
        # meaning, finding the m and b, in y=mx+b

        m, b = utils.math.calc_m_and_b(left_eye, right_eye)
        dist = math.dist(left_eye, right_eye)
        enlarged_dist_wanted = dist * 0.4
        left_point = (int(left_eye[0] + enlarged_dist_wanted),
                      int(m * (left_eye[0] + enlarged_dist_wanted) + b))
        right_point = (int(right_eye[0] - enlarged_dist_wanted),
                       int(m * (right_eye[0] - enlarged_dist_wanted) + b))

        # slope of the perpendicular line
        per_slope = -1.0 / m
        b_left = left_point[1] - per_slope * left_point[0]
        b_right = right_point[1] - per_slope * right_point[0]
        y_margin_wanted = dist / 3

        bottom_right_y = left_point[1] + y_margin_wanted
        bottom_right = (
            int(utils.math.utils.math.x_from_m_b_y(per_slope, b_left, bottom_right_y)), int(bottom_right_y))

        upper_right_y = left_point[1] - y_margin_wanted
        upper_right = (
            int(utils.math.x_from_m_b_y(per_slope, b_left, upper_right_y)), int(upper_right_y))

        upper_left_y = right_point[1] - y_margin_wanted
        upper_left = (
            int(utils.math.x_from_m_b_y(per_slope, b_right, upper_left_y)), int(upper_left_y))
        # upper_left = (int((upper_left_y - b_right) / per_slope), int(upper_left_y))

        bottom_left_y = right_point[1] + y_margin_wanted
        bottom_left = (int(utils.math.x_from_m_b_y(per_slope, b_right,
                                                   bottom_left_y)), int(bottom_left_y))

        # Censor eyes
        cv2.fillConvexPoly(frame, np.array(
            [upper_left, bottom_left, bottom_right, upper_right]), (0, 0, 0))
