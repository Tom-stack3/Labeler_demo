# Labeler demo
## Running instructions
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

An example for an image generated and saved locally by the script:

<img src="https://user-images.githubusercontent.com/76645845/131883453-54ada672-ac1e-4da4-9ea9-47ae5e9dc893.jpg" height="265">

<sup>*The example was generated from [this](http://cdn9.dissolve.com/p/D18_240_012/D18_240_012_0004_600.jpg) image.*</sup>

## Performance
Between capture to capture, the frame freezes, meaning the script is proccessing the image in the background. It usually takes less than 1 second to proccess the image, but it depends on your device. For some it might take 1-3 seconds.
