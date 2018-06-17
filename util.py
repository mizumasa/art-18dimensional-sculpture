# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from six.moves import range

import os
import glob
from PIL import Image
import numpy as np

import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.utils.save as save

import cv2
from PIL import ImageEnhance


def makeOutput(filename,size):
    img = Image.open(filename)
    
    if 1:
        contrast_converter = ImageEnhance.Contrast(img)
        img = contrast_converter.enhance(2.)

    if img.size != (size,size):
        print "resize"
        img = img.resize((size,size))
    imgary = np.asarray(img).copy()

    if 0:
        imgary[:,:,0] = 0

    ary = np.zeros((1,3,size,size))
    for i in range(3):
        ary[0][i] = imgary[:,:,i] / 255.
    return ary

def makeOutputFromFrame(frame,size):
    imgary = cv2.resize(frame,(size,size))
    ary = np.zeros((1,3,size,size))
    for i in range(3):
        ary[0][i] = imgary[:,:,i] / 255.
    return ary

def makeOutputRaw(imgary,size):
    ary = np.zeros((1,3,size,size))
    for i in range(3):
        ary[0][i] = imgary[:,:,i] / 255.
    return ary

def makeInput(size):
    ary = np.zeros((1,3,size,size))
    ary[0][2][:,:] = 1.0
    for i in range(size):
        ary[0][0][i,:] = i * 1. / size
        ary[0][1][:,i] = i * 1. / size
    return ary

def makePng(ary):
    return Image.fromarray(makeBGR(ary))

def makeBGR(ary):
    h = ary.shape[2]
    w = ary.shape[3]
    output = np.zeros((h,w,3),dtype="uint8")
    for i in range(3):
        output[:,:,i] = np.uint8(ary[0,i,:,:]*255)
    return output


if __name__ == '__main__':
    monitor_path = './'
