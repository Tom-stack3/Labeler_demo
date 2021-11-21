import time
import os
import matplotlib.pyplot as plt
import cv2

class Logger:
    def __init__(self, img_extension) -> None:
        self.img_extension = img_extension

        
    def save_to_log(self, img_name, keypoints, distances_and_angles):
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
        row = img_name + self.img_extension + ','
        for k in keypoints:
            if k is None:
                row += "None,"
            else:
                row += '(' + ' '.join(map(str, k)) + '),'
        row += ','.join(map(str, distances_and_angles)) + '\n'

        f.write(str(row))


    def save_img(self, frame, type: str, name: str) -> None:
        if not os.path.exists("output"):
            os.mkdir('output')

        # If it is an image with points and lines drawn on
        if type == "DRAWN":
            save_path = os.path.join("output", name + self.img_extension)
        elif type == "DEPTH":
            save_path = os.path.join(
                "output", name + "_depth" + self.img_extension)
        else:
            save_path = os.path.join("output", name + "_b" + self.img_extension)

        if type == "DEPTH":
            plt.imsave(save_path, frame)
        else:
            cv2.imwrite(save_path, frame)