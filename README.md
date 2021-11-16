# Labeler demo
## Installing and Running
```shell
git lfs install
git clone https://github.com/Tom-stack3/Labeler_demo.git
cd Labeler_demo
pip install -r requirements.txt
python labeler.py
```
- Make sure you have a working camera connected to your device.
- If the command `pip install -r requirements.txt` isn't working and the shell prints: `'pip' is not recognized..`,\
  use instead:
  ```shell
  python -m pip install -r requirements.txt
  ```

## Output
All the labeled images and a log file are saved locally in the `output` folder, which will be created automatically.
In order to protect the user's privacy, the person's eyes detected in a captured image are **censored**, before the image is saved locally.

An example for three images generated from a camera capture, which are then saved locally by the script:

<img alt="points detected drawn" src="https://user-images.githubusercontent.com/76645845/131883453-54ada672-ac1e-4da4-9ea9-47ae5e9dc893.jpg" height="265">
<img alt="depth detected" src="https://user-images.githubusercontent.com/76645845/135473811-0c293fdd-b76e-4493-8fa6-d84e5baeaade.jpg" height="265">
<img alt="original image with censored eyes" src="https://user-images.githubusercontent.com/76645845/135473882-9bcda4d3-2045-4ca9-ac99-f7b04ac103a0.jpg" height="265">

<sup>*The example was generated from [this](http://cdn9.dissolve.com/p/D18_240_012/D18_240_012_0004_600.jpg) image.*</sup>

## Performance
Between capture to capture, the frame freezes, meaning the script is proccessing the image in the background. It usually takes less than 4 second in total to proccess the image, but it depends on your device. Usually the most time-consuming part is the depth analyzer.

## Acknowledgement
- The point detection uses a 2D Pose Estimation model, published in this [paper](https://arxiv.org/pdf/1611.08050.pdf).
- The depth analysis uses the [MiDaS](https://github.com/isl-org/MiDaS) model.
