import os
import cv2
import time
import math
import PySimpleGUI as sg

# calculations for eye censoring
from numpy import ones, vstack, array
from numpy.linalg import lstsq

# for the depth module MiDaS
import torch
import matplotlib.pyplot as plt

CENSOR_EYES = True
RESIZE_FRAME = False

# Camera Settings
CAMERA_WIDTH = 640  # 320 # 480 # 640 # 1024 # 1280
CAMERA_HEIGHT = 480  # 240 # 320 # 480 # 780  # 960

# Output Settings
IMG_OUTPUT_EXTENSION = ".jpg"

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


def calc_angle(a, b, c):
    """
    :param a: a[x, y]
    :param b: b[x, y]
    :param c: c[x, y]
    :return: angle between ab and bc
    """
    if b in (a, c):
        raise ValueError("Undefined angle, two identical points", (a, b, c))

    ang = math.degrees(
        math.atan2(a[1] - b[1], a[0] - b[0]) - math.atan2(c[1] - b[1], c[0] - b[0]))
    return ang + 360 if ang < 0 else ang


def rescale_frame(frame, percent=80):
    max_width = 1000
    max_height = 1000
    if frame.shape[1] <= max_width and frame.shape[0] <= max_height:
        return frame
    else:
        width = int(frame.shape[1] * percent / 100)
        height = int(frame.shape[0] * percent / 100)
        dim = (width, height)
        resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        return rescale_frame(resized)


def calc_m_and_b(point1, point2):
    """
    calculate the slope intercept form of the line from Point 1 to Point 2.
    meaning, finding the m and b, in y=mx+b.
    :param point1: point 1
    :param point2: point 2
    :return: m, b
    """
    points = [point1, point2]
    x_coords, y_coords = zip(*points)
    A = vstack([x_coords, ones(len(x_coords))]).T

    return lstsq(A, y_coords, rcond=None)[0]


def y_from_m_b_x(m, b, x):
    """
    get y from y=mx+b
    :param m: slope (m)
    :param b: b
    :param x: x
    :return: y from y=mx+b
    """
    return m * x + b


def x_from_m_b_y(m, b, y):
    """
    get x from y=mx+b
    :param m: slope (m)
    :param b: b
    :param y: y
    :return: get x from y=mx+b
    """
    return (y - b) / m


def censor_eyes(frame, left_eye, right_eye):
    """
    censor the eyes of the client
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

    m, b = calc_m_and_b(left_eye, right_eye)
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
        int(x_from_m_b_y(per_slope, b_left, bottom_right_y)), int(bottom_right_y))

    upper_right_y = left_point[1] - y_margin_wanted
    upper_right = (
        int(x_from_m_b_y(per_slope, b_left, upper_right_y)), int(upper_right_y))

    upper_left_y = right_point[1] - y_margin_wanted
    upper_left = (
        int(x_from_m_b_y(per_slope, b_right, upper_left_y)), int(upper_left_y))
    # upper_left = (int((upper_left_y - b_right) / per_slope), int(upper_left_y))

    bottom_left_y = right_point[1] + y_margin_wanted
    bottom_left = (int(x_from_m_b_y(per_slope, b_right,
                                    bottom_left_y)), int(bottom_left_y))

    # draw the edges of the censoring polygon:
    # cv2.circle(frame, upper_right, 8, (0, 0, 255))
    # cv2.circle(frame, bottom_right, 8, (0, 0, 255))
    # cv2.circle(frame, upper_left, 8, (0, 0, 255))
    # cv2.circle(frame, bottom_left, 8, (0, 0, 255))

    # Censor eyes
    cv2.fillConvexPoly(frame, array(
        [upper_left, bottom_left, bottom_right, upper_right]), (0, 0, 0))


MODEL_TYPE = "d"


class depthAnalyzer:
    def __init__(self) -> None:
        # 5 seconds in average
        # self.model_type = "DPT_Large" # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        # 2.4 seconds in average
        # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
        self.model_type = "DPT_Hybrid"
        # 0.8 seconds in average
        # self.model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

        global MODEL_TYPE
        MODEL_TYPE = self.model_type

        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.midas.to(self.device)
        self.midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if self.model_type == "DPT_Large" or self.model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def calc_depth(self, frame, img_name):
        input_batch = self.transform(frame).to(self.device)

        # Predict and resize to original resolution
        with torch.no_grad():
            prediction = self.midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=True,
            ).squeeze()

        output = prediction.cpu().numpy()

        # Show and save result
        save_img(output, "DEPTH", img_name)


def label_image(net, frame, dpa: depthAnalyzer, need_to_show_frame=True):
    nPoints = 18
    POSE_PAIRS = [[NOSE, NECK], [NECK, RSH], [NECK, LSH], [NOSE, RSH], [NOSE, LSH], [RSH, LSH], [REYE, NOSE],
                  [NOSE, LEYE]]  # , [REAR, REYE], [LEYE, LEAR]]

    print('Original Dimensions : ', frame.shape)

    if RESIZE_FRAME:
        scale_percent = 60  # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)

        # Resize image
        resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        print('Resized Dimensions : ', resized.shape)
        frame = resized

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    threshold = 0.1

    t = time.time()
    # input image dimensions for the network
    inWidth = 368
    inHeight = 368
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                    (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)

    output = net.forward()
    print("time taken by network : {:.3f}".format(time.time() - t))

    H = output.shape[2]
    W = output.shape[3]

    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # Confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold:
            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else:
            points.append(None)

    original_frame = frame

    if CENSOR_EYES:
        # Censor the eyes
        censor_eyes(frame, points[LEYE], points[REYE])

    # Save image without the points and lines detected drawn on.
    img_name = save_img(frame, "EMPTY")

    # Calc depth
    dt = time.time()
    dpa.calc_depth(original_frame, img_name)
    print("time taken by depth module : {:.3f}".format(time.time() - dt))

    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            # we do all this, to print the nose in pink, the Neck in green, and the shoulders in red.
            # we work with BGR format.
            cv2.circle(frame, points[partA], 8, (0, 0, 255),
                       thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB], 8, (0, 0, 255),
                       thickness=-1, lineType=cv2.FILLED)
            if 0 == pair[0]:
                cv2.circle(
                    frame, points[partA], 8, (255, 0, 255), thickness=-1, lineType=cv2.FILLED)
            elif 0 == pair[1]:
                cv2.circle(
                    frame, points[partB], 8, (255, 0, 255), thickness=-1, lineType=cv2.FILLED)
            if 1 == pair[0]:
                cv2.circle(frame, points[partA], 8, (0, 255, 0),
                           thickness=-1, lineType=cv2.FILLED)
            elif 1 == pair[1]:
                cv2.circle(frame, points[partB], 8, (0, 255, 0),
                           thickness=-1, lineType=cv2.FILLED)

            # we draw the connecting line
            cv2.line(frame, points[partA], points[partB],
                     (0, 255, 255), 3, lineType=cv2.LINE_AA)

    if need_to_show_frame:
        cv2.imshow('Output-Skeleton', frame)

    save_img(frame, "DRAWN", img_name)

    keypoints, distances_and_angles = calc_data_for_log(points)
    save_to_log(img_name, keypoints, distances_and_angles)

    print("Saved Image!")
    print("Total time taken : {:.3f}".format(time.time() - t))
    print()
    cv2.waitKey(0)


def calc_data_for_log(points):
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
            angles.append(calc_angle(
                points[triplets[0]], points[triplets[1]], points[triplets[2]]))
        else:
            angles.append(None)

    return keypoints, distances + angles


def save_to_log(img_name, keypoints, distances_and_angles):
    """
    save data to log file
    :param img_name: image name
    :param keypoints: keypoints detected
    :param distances_and_angles: distances and angles calculated
    :return: None
    """
    LOG_FIRST_ROW = "File name,Nose,Neck,Right Shoulder,Left Shoulder,Right Eye,Left Eye," \
                    "Nose <-> Neck,Neck <-> RSH,Neck <-> LSH,RSH <-> LSH,Nose <-> RSH,Nose <-> LSH" \
                    "(Nose <-> Neck <-> RSH),(LSH <-> Neck <-> Nose)"

    try:
        open("output/log.csv")
    except IOError:
        # If file doesn't exist, we create one
        f = open("output/log.csv", "a")
        f.write(LOG_FIRST_ROW + '\n')
    f = open("output/log.csv", "a")
    row = img_name + IMG_OUTPUT_EXTENSION + ','
    for k in keypoints:
        if k is None:
            row += "None,"
        else:
            row += '(' + ' '.join(map(str, k)) + '),'
    row += ','.join(map(str, distances_and_angles)) + '\n'

    f.write(str(row))


def save_img(frame, type: str, name=None):
    if name is None:
        name = time.strftime(
            MODEL_TYPE+' ' + '%Y_%m_%d %H_%M_%S', time.localtime(time.time()))

    if not os.path.exists("output"):
        os.mkdir('output')

    # If it is an image with points and lines drawn on
    if type == "DRAWN":
        save_path = os.path.join("output", name + IMG_OUTPUT_EXTENSION)
    elif type == "DEPTH":
        save_path = os.path.join(
            "output", name + "_depth" + IMG_OUTPUT_EXTENSION)
    else:
        save_path = os.path.join("output", name + "_b" + IMG_OUTPUT_EXTENSION)

    if type == "DEPTH":
        plt.imsave(save_path, frame)
    else:
        cv2.imwrite(save_path, frame)
    return name


def main():
    # Load the Model
    protoFile = "coco/pose_deploy_linevec.prototxt"
    weightsFile = "coco/pose_iter_440000.caffemodel"
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)

    dpa = depthAnalyzer()

    # Camera Settings
    frameSize = (CAMERA_WIDTH, CAMERA_HEIGHT)
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    time.sleep(1.0)

    # Init Windows Manager
    sg.theme("DarkBlue")

    # Capture button logo
    img_path = "img/camera.png"

    capture_btn = sg.Button(button_color=sg.TRANSPARENT_BUTTON,
                            image_filename=img_path, image_size=(50, 50), image_subsample=2,
                            border_width=0, key="_capture_")

    # Def webcam col
    colwebcam1_layout = [[sg.Text("Camera View", size=(60, 1), justification="center")],
                         [sg.Image(filename="", key="cam1")], [capture_btn]]
    colwebcam1 = sg.Column(colwebcam1_layout, element_justification='center')

    colslayout = [colwebcam1]

    layout = [colslayout]

    window = sg.Window("Labeler demo", layout,
                       no_titlebar=False, alpha_channel=1, grab_anywhere=False,
                       return_keyboard_events=True, location=(100, 100), icon="img/camera.ico")

    while True:
        event, values = window.read(timeout=20)

        if event == sg.WIN_CLOSED:
            break

        # Get camera frame
        ret, frameOrig = video_capture.read()
        frame = cv2.resize(frameOrig, frameSize)

        # If you want to procces an image not from the camera
        # frame = cv2.imread("path_to_image")

        # Update webcam
        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        window["cam1"].update(data=imgbytes)

        # If the capture button was pressed
        if event == "_capture_":
            label_image(net, frame, dpa, False)

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
