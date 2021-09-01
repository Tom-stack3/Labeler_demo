import os
import cv2
import time
import math
import PySimpleGUI as sg

# calculations for eye censoring
from numpy import ones, vstack, array
from numpy.linalg import lstsq

CENSOR_EYES = True

'''
COCO Output Format:
 * Nose – 0, Neck – 1, Right Shoulder – 2, Right Elbow – 3, Right Wrist – 4, Left Shoulder – 5, Left Elbow – 6, 
 * Left Wrist – 7, Right Hip – 8, Right Knee – 9, Right Ankle – 10, Left Hip – 11, Left Knee – 12, LAnkle – 13,
 * Right Eye – 14, Left Eye – 15, Right Ear – 16, Left Ear – 17, Background – 18
'''

'''
* lsh = left shoulder
* rsh = right shoulder
* reye = right eye
* leye = left eye
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

    # calculate the slope intercept form of the line from the Left Eye to the Right eye
    # meaning, finding the m and b, in y=mx+b

    m, b = calc_m_and_b(left_eye, right_eye)
    dist = math.dist(left_eye, right_eye)
    enlarged_dist_wanted = dist * 0.4
    left_point = (int(left_eye[0] + enlarged_dist_wanted), int(m * (left_eye[0] + enlarged_dist_wanted) + b))
    right_point = (int(right_eye[0] - enlarged_dist_wanted), int(m * (right_eye[0] - enlarged_dist_wanted) + b))

    # slope of the perpendicular line
    per_slope = -1.0 / m
    b_left = left_point[1] - per_slope * left_point[0]
    b_right = right_point[1] - per_slope * right_point[0]
    y_margin_wanted = dist / 3

    bottom_right_y = left_point[1] + y_margin_wanted
    bottom_right = (int(x_from_m_b_y(per_slope, b_left, bottom_right_y)), int(bottom_right_y))

    upper_right_y = left_point[1] - y_margin_wanted
    upper_right = (int(x_from_m_b_y(per_slope, b_left, upper_right_y)), int(upper_right_y))

    upper_left_y = right_point[1] - y_margin_wanted
    upper_left = (int(x_from_m_b_y(per_slope, b_right, upper_left_y)), int(upper_left_y))
    # upper_left = (int((upper_left_y - b_right) / per_slope), int(upper_left_y))

    bottom_left_y = right_point[1] + y_margin_wanted
    bottom_left = (int(x_from_m_b_y(per_slope, b_right, bottom_left_y)), int(bottom_left_y))

    cv2.circle(frame, upper_right, 8, (0, 0, 255))
    cv2.circle(frame, bottom_right, 8, (0, 0, 255))

    cv2.circle(frame, upper_left, 8, (0, 0, 255))
    cv2.circle(frame, bottom_left, 8, (0, 0, 255))

    # Censor eyes
    cv2.fillConvexPoly(frame, array([upper_left, bottom_left, bottom_right, upper_right]), (0, 0, 0))


def label_image(net, frame, need_to_show_frame=True):
    nPoints = 18
    POSE_PAIRS = [[NOSE, NECK], [NECK, RSH], [NECK, LSH], [NOSE, RSH], [NOSE, LSH], [RSH, LSH], [REYE, NOSE],
                  [NOSE, LEYE]]  # , [REAR, REYE], [LEYE, LEAR]]

    print('Original Dimensions : ', frame.shape)

    scale_percent = 60  # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
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
        # confidence map of corresponding body's part.
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

    if CENSOR_EYES:
        # Censor the eyes
        censor_eyes(frame, points[LEYE], points[REYE])

    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            # we do all this, to print the nose in pink, the neck in green, and the shoulders in red.
            # we work with BGR format.
            cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            if 0 == pair[0]:
                cv2.circle(frame, points[partA], 8, (255, 0, 255), thickness=-1, lineType=cv2.FILLED)
            elif 0 == pair[1]:
                cv2.circle(frame, points[partB], 8, (255, 0, 255), thickness=-1, lineType=cv2.FILLED)
            if 1 == pair[0]:
                cv2.circle(frame, points[partA], 8, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
            elif 1 == pair[1]:
                cv2.circle(frame, points[partB], 8, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)

            # we draw the connecting line
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)

    if need_to_show_frame:
        cv2.imshow('Output-Skeleton', frame)

    save_img(frame)

    print("Captured Image!")
    print("Total time taken : {:.3f}".format(time.time() - t))

    cv2.waitKey(0)


def save_img(frame):
    current_time = time.strftime('%Y_%m_%d %H_%M_%S', time.localtime(time.time()))

    if not os.path.exists("output"):
        os.mkdir('output')

    save_path = os.path.join("output", current_time + ".jpg")
    cv2.imwrite(save_path, frame)


def main():
    # Load the Model
    protoFile = "coco/pose_deploy_linevec.prototxt"
    weightsFile = "coco/pose_iter_440000.caffemodel"
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)

    # Camera Settings
    camera_width = 1024  # 320 # 480 # 640 # 1024 # 1280
    camera_height = 780  # 240 # 320 # 480 # 780  # 960
    frameSize = (camera_width, camera_height)
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    time.sleep(1.0)

    # init Windows Manager
    sg.theme("DarkBlue")

    # Capture button logo
    img_path = "img/camera.png"

    capture_btn = sg.Button(button_color=sg.TRANSPARENT_BUTTON,
                            image_filename=img_path, image_size=(50, 50), image_subsample=2,
                            border_width=0, key="_capture_")

    # def webcam col
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

        # get camera frame
        ret, frameOrig = video_capture.read()
        frame = cv2.resize(frameOrig, frameSize)

        # update webcam
        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        window["cam1"].update(data=imgbytes)

        # if the capture button was pressed
        if event == "_capture_":
            label_image(net, frame, False)

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
