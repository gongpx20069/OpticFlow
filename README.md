# OpticFlow
This is a code base for optical flow estimation using Python language, integrating various methods of OpenCV.
# Dependency Library

- Opencv: ```pip3 install opencv-python```
- opencv-contrib: ```pip3 install opencv-contrib-python==3.3.0.10```

# How to use the script

What if we need to capture the dense optical flow between two images? We only need the following code:

```python
import cv2
from opencv_flow import OpticFlow

of = OpticFLow(mode = 'pcaflow')

img1 = cv2.imread('test/1.jpg')
img2 = cv2.imread('test/2.jpg')
flow = of.getflow(img1, img2)
```

You can check the [wiki docs](https://github.com/gongpx20069/OpticFlow/wiki/Quick-start) for specific methods!

# Compatible with Pytorch

In the quick start section, all of our data structures are based on numpy. In this chapter, we mainly make the data structure compatible with Pytorch and develop more functions.
You can check the [wiki docs](https://github.com/gongpx20069/OpticFlow/wiki/Compatible-with-Pytorch) for specific methods!
