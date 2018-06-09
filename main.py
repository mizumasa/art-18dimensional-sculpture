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

from args import get_args
import iterator

import cv2
from PIL import ImageEnhance

SIZE = 200

VIEW_ON = False
VIEW_ON = True

def network(x, maxh=16, depth=8):
    with nn.parameter_scope("net"):
        # (1, 28, 28) --> (32, 16, 16)
        with nn.parameter_scope("convIn"):
            out = F.tanh(PF.convolution(x, maxh, (1, 1), with_bias=False))
        for i in range(depth):
            with nn.parameter_scope("conv"+str(i)):
                out = F.tanh(PF.convolution(out, maxh, (1, 1), with_bias=False))
        with nn.parameter_scope("convOut"):
            out = F.sigmoid(PF.convolution(out, 3, (1, 1), with_bias=False))
    return out

def makeOutput(filename):
    img = Image.open(filename)
    if img.size != (SIZE,SIZE):
        print "resize"
        img = img.resize((SIZE,SIZE))
    imgary = np.asarray(img)
    ary = np.zeros((1,3,SIZE,SIZE))
    for i in range(3):
        ary[0][i] = imgary[:,:,i] / 255.
    return ary

def makeOutputFromFrame(frame):
    imgary = cv2.resize(frame,(SIZE,SIZE))
    ary = np.zeros((1,3,SIZE,SIZE))
    for i in range(3):
        ary[0][i] = imgary[:,:,i] / 255.
    return ary

def makeInput():
    ary = np.zeros((1,3,SIZE,SIZE))
    ary[0][2][:,:] = 1.0
    for i in range(SIZE):
        ary[0][0][i,:] = i * 1. / SIZE
        ary[0][1][:,i] = i * 1. / SIZE
    return ary

def makePng(ary):
    output = np.zeros((SIZE,SIZE,3),dtype="uint8")
    for i in range(3):
        output[:,:,i] = np.uint8(ary[0,i,:,:]*255)
    return Image.fromarray(output)

def makeBGR(ary):
    output = np.zeros((SIZE,SIZE,3),dtype="uint8")
    for i in range(3):
        output[:,:,i] = np.uint8(ary[0,i,:,:]*255)
    return output

def makeBGRVstack(ary):
    output = np.zeros((SIZE*2,SIZE,3),dtype="uint8")
    for i in range(3):
        output[:,:,i] = np.uint8(ary[0,i,:,:]*255)
    return output

def train(args):
    if VIEW_ON:
        cv2.namedWindow('screen', cv2.WINDOW_NORMAL)
        #cv2.setWindowProperty('screen', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    from nnabla.contrib.context import extension_context
    extension_module = args.context
    if args.context is None:
        extension_module = 'cpu'
    logger.info("Running in %s" % extension_module)
    ctx = extension_context(extension_module, device_id=args.device_id)
    nn.set_default_context(ctx)

    x = nn.Variable([1, 3, SIZE, SIZE])
    #y = network(x, maxh=8, depth=5)
    y = network(x)
    
    dataIn = makeInput()
    output = nn.Variable([1, 3, SIZE, SIZE])
    
    dataOut = makeOutput("test.png")
    output.d = dataOut

    loss = F.mean(F.squared_error(y, output))

    param = nn.get_parameters()
    for i,j in param.items():
        param.get(i).d = np.random.randn(*(j.d.shape))

    solver = S.Adam(args.learning_rate, beta1=0.5)
    with nn.parameter_scope("net"):
        solver.set_parameters(nn.get_parameters())

    if VIEW_ON:
        cap = cv2.VideoCapture(0)
    # Training loop.
    count = 0
    while 1:
        if VIEW_ON:
            ret, frame = cap.read()
        else:
            ret, frame = (True,np.zeros((720,1280,3),dtype="uint8"))
        if ret:
            contrast_converter = ImageEnhance.Contrast(Image.fromarray(frame))
            frame = np.asarray(contrast_converter.enhance(2.))
            output.d = makeOutputFromFrame(frame)
        count += 1
        if count % 30 == 0:
            print count
        x.d = dataIn.copy()
        solver.zero_grad()
        loss.forward(clear_no_need_grad=True)

        #cv2.imshow('screen', makeBGR(y.d))
        if VIEW_ON:
            cv2.imshow('screen', makeBGRVstack(np.concatenate([y.d, output.d], axis=2)))

            k = cv2.waitKey(10)
            if k == 27: #ESC
                break
            else:
                print k
        if 0 and count % 10 == 0:
            img = makePng(y.d)
            img.save(os.path.join(args.model_save_path, "output_%06d.png" % count))
        loss.backward(clear_buffer=True)
        solver.weight_decay(args.weight_decay)
        solver.update()

    return


if __name__ == '__main__':
    monitor_path = './'
    args = get_args(monitor_path=monitor_path, model_save_path=monitor_path,
                    max_iter=20000, learning_rate=0.002, batch_size=64,
                    weight_decay=0.0001)
    train(args)
