Chainer Grad-CAM
====

Experimental Chainer implementation of Grad-CAM without modifying network structure.

# Caution

This implementation depends on Chainer internal implementation, so it may not work on later version of Chainer.

# Requirements

* Chainer 3.0
* OpenCV

# Sample Usage

## import package

```
from lib import gradcam
```

## Make Chainer neural network model

```
from chainercv.links import VGG16

model = VGG16(pretrained_model='imagenet')
```

## Load image and make input array

```
image = cv2.imread(image_path)
image_size = (224, 224)
image = cv2.resize(image, image_size)

x = np.float32(image)
x = x.transpose((2, 0, 1))[np.newaxis,::-1,:,:]
x -= model.mean
```

## Make gradient for model output

```
y_grad = np.zeros((1, 1000), dtype=np.float32)
y_grad[0, args.label] = 1.0
```

## Calculate Grad-CAM

```
gcam = gradcam.gradcam(model, x, [model.conv5_3.conv, F.relu], y_grad=y_grad)
```

## Make Grad-CAM heatmap

```
gcam = gcam[0]
heatmap_image = gradcam.heatmap(gcam, image_size)
cv2.imwrite(heatmap_path, heatmap_image)
```

## Make Grad-CAM heatmap over input image

```
overlay_image = gradcam.overlay(image, gcam)
cv2.imwrite(overlay_path, overlay_image)
```

# Sample program usage

```
$ python src/vgg16.py image_path heatmap_path overlay_path label [-g gpu_device]
```

Parameters

* `image_path`: (Required) Input image file path
* `heatmap_path`: (Required) File path for heatmap image
* `overlay_path`: (Required) File path for heatmap over input image
* `label`: (Required) Class label index
* `gpu_device`: (Optional) GPU device ID, negative value indicates CPU (default: -1)
