from PySimpleGUI.PySimpleGUI import No
import yaml
import cv2
import utils.math
import math
import numpy as np
import time

'''
COCO Output Format:
 * Nose – 0, Neck – 1, Right Shoulder – 2, Right Elbow – 3, Right Wrist – 4, Left Shoulder – 5, Left Elbow – 6, 
 * Left Wrist – 7, Right Hip – 8, Right Knee – 9, Right Ankle – 10, Left Hip – 11, Left Knee – 12, LAnkle – 13,
 * Right Eye – 14, Left Eye – 15, Right Ear – 16, Left Ear – 17, Background – 18
'''

'''
* LSH = left shoulder
* RSH = right shoulder
* REye = right eye
* LEye = left eye
'''

NOSE = 0
NECK = 1
RSH = 2
LSH = 5
REYE = 14
LEYE = 15
REAR = 16
LEAR = 17


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

        self.nPoints = 18
        self.POSE_PAIRS = [[NOSE, NECK], [NECK, RSH], [NECK, LSH], [NOSE, RSH], [NOSE, LSH], [RSH, LSH], [REYE, NOSE],
                           [NOSE, LEYE]]  # , [REAR, REYE], [LEYE, LEAR]]

        # Empty list to store the current detected keypoints
        self.points_detected = []

    def analyze_image(self, frame, img_name):
        # save the frame
        self.frame = frame
        self.img_name = img_name

        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        threshold = 0.1

        t = time.time()

        # input image dimensions for the network
        inWidth = 368
        inHeight = 368
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                        (0, 0, 0), swapRB=False, crop=False)

        self.net.setInput(inpBlob)

        output = self.net.forward()
        print("time taken by network : {:.3f}".format(time.time() - t))

        H = output.shape[2]
        W = output.shape[3]

        for i in range(self.nPoints):
            # Confidence map of corresponding body's part.
            probMap = output[0, i, :, :]

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # Scale the point to fit on the original image
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H

            if prob > threshold:
                # Add the point to the list if the probability is greater than the threshold
                self.points_detected.append((int(x), int(y)))
            else:
                self.points_detected.append(None)

    def draw_points(self):
        # Draw Skeleton
        for pair in self.POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]

            if self.points_detected[partA] and self.points_detected[partB]:
                # Do all this, to print the nose in pink, the Neck in green, and the shoulders in red.
                # We work with BGR format.
                cv2.circle(self.frame, self.points_detected[partA], 8, (0, 0, 255),
                        thickness=-1, lineType=cv2.FILLED)
                cv2.circle(self.frame, self.points_detected[partB], 8, (0, 0, 255),
                        thickness=-1, lineType=cv2.FILLED)
                if 0 == pair[0]:
                    cv2.circle(
                        self.frame, self.points_detected[partA], 8, (255, 0, 255), thickness=-1, lineType=cv2.FILLED)
                elif 0 == pair[1]:
                    cv2.circle(
                        self.frame, self.points_detected[partB], 8, (255, 0, 255), thickness=-1, lineType=cv2.FILLED)
                if 1 == pair[0]:
                    cv2.circle(self.frame, self.points_detected[partA], 8, (0, 255, 0),
                            thickness=-1, lineType=cv2.FILLED)
                elif 1 == pair[1]:
                    cv2.circle(self.frame, self.points_detected[partB], 8, (0, 255, 0),
                            thickness=-1, lineType=cv2.FILLED)

                # Draw the connecting line
                cv2.line(self.frame, self.points_detected[partA], self.points_detected[partB],
                        (0, 255, 255), 3, lineType=cv2.LINE_AA)


    def save_img(self, type: str):
        self.logger.save_img(self.frame, type, self.img_name)

    @staticmethod
    def __calc_data_for_log(points):
        # Only the points we care about
        keypoints = [points[i] for i in range(len(points)) if i in [
            NOSE, NECK, RSH, LSH, REYE, LEYE]]
        distances_wanted = [(NOSE, NECK), (NECK, RSH), (NECK, LSH),
                            (RSH, LSH), (NOSE, RSH), (NOSE, LSH)]
        angles_wanted = [(NOSE, NECK, RSH), (LSH, NECK, NOSE)]

        distances = []
        angles = []

        for pair in distances_wanted:
            # If both in pair were detected, we add the distance between them. else, we add None
            if points[pair[0]] is not None and points[pair[1]] is not None:
                distances.append(math.dist(points[pair[0]], points[pair[1]]))
            else:
                distances.append(None)
        for triplets in angles_wanted:
            # If all were detected, we add the angle between them. else, we add None
            if points[triplets[0]] is not None and points[triplets[1]] is not None and points[triplets[2]] is not None:
                angles.append(utils.math.calc_angle(
                    points[triplets[0]], points[triplets[1]], points[triplets[2]]))
            else:
                angles.append(None)

        return keypoints, distances + angles

    def log_img_info(self) -> None:
        keypoints, distances_and_angles = PointsAnalyzer.calc_data_for_log(self.points)
        self.logger.save_to_log(self.img_name, keypoints, distances_and_angles)

    def censor_eyes(self) -> None:
        """
        censor the eyes of the person in the current frame
        :return: None
        """
        # If there are no points detected yet
        if len(self.points_detected) == 0:
            return

        left_eye = self.points_detected[LEYE]
        right_eye = self.points_detected[REYE]

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
        cv2.fillConvexPoly(self.frame, np.array(
            [upper_left, bottom_left, bottom_right, upper_right]), (0, 0, 0))
