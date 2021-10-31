# Characters Segmentation and Recognition for Vehicle License Plate
Characters segmentation and recognition using OpenCV and deep learning

## Prerequisites

<ul>
<li>Python 3.5.2</li>
<li>Keras 2.2.4</li>
<li>Tensorflow 1.13.1</li>
<li>OpenCV 4.4.0</li>
<li>Numpy 1.16.2</li>
</ul>

## Usage

The pre-trained deep learning model used in this project is taken from my other project https://github.com/minhthangdang/CharactersRecognition. The model is too big for github, so you can download it from https://drive.google.com/file/d/1ojsMc-VYSFKromwSm4Qx3kRpPmi2klOV/view?usp=sharing. Alternatively you can train it yourself from scratch using my repository above.

Copy the pre-trained weights into the same folder of this project.

To test on an image, just run:

```python
python plate_read.py --model characters_model.weights --image plate1.jpg
```

Where <i>characters_model.weights</i> is the pre-trained model and <i>plate1.jpg</i> is the test image.

The result will look like this:<br>
<img src="https://raw.githubusercontent.com/minhthangdang/minhthangdang.github.io/master/plate05.PNG" width="300" alt="Characters segmentation and recognition for vehicle license plate" title="Characters segmentation and recognition">

Demo video:
[![Demo Characters Segmentation and Recognition](https://img.youtube.com/vi/r_09uCdiH1Y/hqdefault.jpg)](https://www.youtube.com/watch?v=r_09uCdiH1Y)

Full tutorial is available at http://dangminhthang.com/computer-vision/characters-segmentation-and-recognition-for-vehicle-license-plate/
