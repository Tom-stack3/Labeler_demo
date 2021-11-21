import cv2
import time
import PySimpleGUI as sg

from labeler import Labeler
import yaml


def main():
    labeler = Labeler()
    # Load settings from yaml
    with open('settings.yaml', 'r') as input_file:
        settings = yaml.load(input_file, yaml.FullLoader)

    # Camera Settings
    frameSize = (settings["camera"]["width"], settings["camera"]["height"])
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    time.sleep(1.0)

    # Init Windows Manager
    sg.theme(settings["window"]["sg_theme"])

    # Capture button logo
    img_path = settings["window"]["button_img_path"]

    capture_btn = sg.Button(button_color=sg.TRANSPARENT_BUTTON,
                            image_filename=img_path, image_size=(50, 50), image_subsample=2,
                            border_width=0, key="_capture_")

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
        # frame = cv2.imread("PATH_TO_IMAGE")

        # Update webcam
        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        window["cam1"].update(data=imgbytes)

        # If the capture button was pressed
        if event == "_capture_":
            labeler.label_image(frame)

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
