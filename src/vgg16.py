import argparse
import numpy as np
import cv2 as cv2

import chainer
from chainer import cuda
from chainer import functions as F
from chainercv.links import VGG16

from lib import gradcam


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path')
    parser.add_argument('heatmap_path')
    parser.add_argument('overlay_path')
    parser.add_argument('label', type=int)
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU device index, negative value indicates CPU' +
                        ' (default: -1)')
    return parser.parse_args()


def main():
    args = parse_args()

    model = VGG16(pretrained_model='imagenet')
    image = cv2.imread(args.image_path)
    image_size = (224, 224)
    image = cv2.resize(image, image_size)

    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
        xp = cuda.cupy
    else:
        xp = np

    x = np.float32(image)
    x = x.transpose((2, 0, 1))[np.newaxis,::-1,:,:]
    x -= model.mean
    x = xp.asarray(x)

    y_grad = xp.zeros((1, 1000), dtype=np.float32)
    y_grad[0, args.label] = 1.0
    gcam = gradcam.gradcam(model, x, [model.conv5_3.conv, F.relu], y_grad=y_grad)
    gcam = cuda.to_cpu(gcam[0])

    heatmap_image = gradcam.heatmap(gcam, image_size)
    cv2.imwrite(args.heatmap_path, heatmap_image)

    overlay_image = gradcam.overlay(image, gcam)
    cv2.imwrite(args.overlay_path, overlay_image)


if __name__ == '__main__':
    main()
