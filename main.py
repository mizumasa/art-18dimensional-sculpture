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

import time
import cv2
from PIL import ImageEnhance

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import util

SIZE = 800

VIEW_ON = False
VIEW_ON = True

LOOP_MODE = not True
LOOP_LENGTH = 60
LOOP_FADE = 30

TRAIN_ON = True
SAVE_ON = True

def network(x, maxh=16, depth=12):
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

class App:
    def __init__(self,args):
        self.args =args
        self.start = time.time()
        # USB camera setup
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened() is False:
            raise("IO Error")
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.initWindowWidth = 800
        self.initWindowHeight = 400
        self.frameStart = time.time()
        self.frameSpentTime = []
        self.canvas = np.zeros((self.initWindowHeight,self.initWindowWidth,3),dtype = "uint8")
        self.lenna = util.makeOutput("lenna.jpg",SIZE)

        self.param = {"pre":{},"next":{}}

        from nnabla.contrib.context import extension_context
        extension_module = args.context
        if args.context is None:
            extension_module = 'cpu'
        logger.info("Running in %s" % extension_module)
        ctx = extension_context(extension_module, device_id=args.device_id)
        nn.set_default_context(ctx)

        self.x = nn.Variable([1, 3, SIZE, SIZE])
        #y = network(x, maxh=8, depth=5)
        self.y = network(self.x)
        
        self.dataIn = util.makeInput(SIZE)
        self.output = nn.Variable([1, 3, SIZE, SIZE])
        
        dataOut = util.makeOutput("test.png",SIZE)
        self.output.d = dataOut

        self.loss = F.mean(F.squared_error(self.y, self.output))

        param = nn.get_parameters()
        for i,j in param.items():
            param.get(i).d = np.random.randn(*(j.d.shape))

        self.solver = S.Adam(args.learning_rate, beta1=0.5)
        with nn.parameter_scope("net"):
            self.solver.set_parameters(nn.get_parameters())
        self.count = 0
        self.countLoop = 0
        self.initParam()

    def draw(self):
        self.frameStart = time.time()
        # Paste into texture to draw at high speed
        ret, frame = self.cap.read() #read camera image
        #ret, frame = (True,np.zeros((720,1280,3),dtype="uint8"))
        #img = cv2.imread('image.png') # if use the image file
        if ret:
            img= cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) #BGR-->RGB
            img_ = img.copy()
            contrast_converter = ImageEnhance.Contrast(Image.fromarray(img))
            img = np.asarray(contrast_converter.enhance(2.))
            #self.output.d = util.makeOutputFromFrame(img,SIZE)
            self.output.d = self.lenna

            if LOOP_MODE:
                self.countLoop += 1
                if self.countLoop == LOOP_LENGTH:
                    self.countLoop = 0
                if self.countLoop < LOOP_FADE:
                    self.setNowParam(1. * self.countLoop / LOOP_FADE)
                if self.countLoop == LOOP_FADE:
                    self.nextParam()

            self.count += 1
            if self.count % 30 == 0:
                print self.count, "fps:", 1. * len(self.frameSpentTime) / sum(self.frameSpentTime) 
                self.frameSpentTime = []
            self.x.d = self.dataIn.copy()
            self.solver.zero_grad()
            self.loss.forward(clear_no_need_grad=True)

            #cv2.imshow('screen', util.makeBGRVstack(np.concatenate([y.d, output.d], axis=2)))
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            #outframe = util.makeBGR(np.concatenate([self.y.d, util.makeOutputFromFrame(img_,SIZE)], axis=2))

            self.startCanvas()
            
            if 1:
                self.drawImage(util.makeBGR(self.y.d),SIZE,0,SIZE,SIZE)
                self.drawImage(util.makeBGR(self.output.d),0,0,SIZE,SIZE)
                #for i in range(16):
                #    for j in range(10):
                #        self.drawImage(util.makeBGR(self.y.d),i*100,j*100,100,100)
                self.drawCanvas()

            if 0:
                self.drawImage(img_,100,0,300,300,False)
                self.drawImage(util.makeBGR(self.y.d),100,200,100,100,False)
                for i in range(16):
                    for j in range(10):
                        self.drawImage(util.makeBGR(self.y.d),i*100,j*100,100,100,False)


            glFlush();
            glutSwapBuffers()

            if SAVE_ON and self.count % 1 == 0:
                img2 = util.makePng(self.y.d)
                img2.save(os.path.join(self.args.model_save_path, "output_%06d.png" % self.count))
            if TRAIN_ON:
                self.loss.backward(clear_buffer=True)
                self.solver.weight_decay(self.args.weight_decay)
                self.solver.update()
        self.frameSpentTime.append(time.time() - self.frameStart)

    def drawImage(self,ary,x,y,w,h,useCanvas = True):
        if useCanvas:
            if (ary.shape == (h,w,3)):
                try:
                    self.canvas[y:y+h,x:x+w,:] = ary
                except:
                    print "WARN out of area",(x,y,w,h)
            else:
                try:
                    self.canvas[y:y+h,x:x+w,:] = cv2.resize(ary,(w,h))
                except:
                    print "WARN out of area",(x,y,w,h)
            return

        h_, w_ = ary.shape[:2]
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w_, h_, 0, GL_RGB, GL_UNSIGNED_BYTE, ary)
        #glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glColor3f(1.0, 1.0, 1.0)
        glEnable(GL_TEXTURE_2D)
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        left = 2. * x / self.windowSizeW - 1.
        top = 2. * y / self.windowSizeH - 1.
        right = 2. * (x+w) / self.windowSizeW - 1.
        bottom = 2. * (y+h) / self.windowSizeH - 1.
        # draw square
        glBegin(GL_QUADS) 
        glTexCoord2d(0.0, 1.0)
        glVertex3d(left, - bottom,  0.0)
        glTexCoord2d(1.0, 1.0)
        glVertex3d(right, - bottom,  0.0)
        glTexCoord2d(1.0, 0.0)
        glVertex3d(right, - top,  0.0)
        glTexCoord2d(0.0, 0.0)
        glVertex3d(left, - top,  0.0)
        glEnd()
        return

    def setNextParam(self):
        param = nn.get_parameters()
        for i,j in param.items():
            param.get(i).d = np.random.randn(*(j.d.shape))
        return

    def initParam(self):
        param = nn.get_parameters()
        for i,j in param.items():
            self.param["pre"][i] = np.random.randn(*(j.d.shape))
            self.param["next"][i] = np.random.randn(*(j.d.shape))
        return

    def nextParam(self):
        for i in self.param["next"].keys():
            self.param["pre"][i] = self.param["next"][i].copy()
            self.param["next"][i] = np.random.randn(*(self.param["next"][i].shape))
        return

    def setNowParam(self,level):#level = 0 -> 1
        param = nn.get_parameters()
        for i,j in param.items():
            param.get(i).d = self.param["next"][i] * level + self.param["pre"][i] * (1. - level)
        return

    def startCanvas(self):
        if self.canvas.shape != (self.windowSizeH,self.windowSizeW,3):
            print "change canvas size"
            self.canvas = np.zeros((self.windowSizeH,self.windowSizeW,3),dtype = "uint8")
        else:
            self.canvas[:,:,:]=0

    def drawCanvas(self):
        self.drawImage(self.canvas,0,0,self.windowSizeW,self.windowSizeH,False)

    def init(self):
        glutFullScreen()
        glClearColor(0.7, 0.7, 0.7, 0.7)

    def idle(self):
        glutPostRedisplay()

    def reshape(self,w, h):
        self.windowSizeW = w
        self.windowSizeH = h
        glViewport(0, 0, w, h)
        glLoadIdentity()
        #Make the display area proportional to the size of the view
        #glOrtho(-w / self.windowWidth, w / self.windowWidth, -h / self.windowHeight, h / self.windowHeight, -1.0, 1.0)

    def keyboard(self,key, x, y):
        # convert byte to str
        key = key.decode('utf-8')
        # press q to exit
        if key == 'q':
            print('exit')
            sys.exit()
        if key == 'n':
            #self.setNextParam()
            self.nextParam()
            self.setNowParam(0.5)
        if key == 't':
            global TRAIN_ON
            TRAIN_ON = not TRAIN_ON
        if key == 's':
            global SAVE_ON
            SAVE_ON = not SAVE_ON
        if key == 'f':
            if self.initWindowWidth == self.windowSizeW:
                glutFullScreen()
            else:
                glutReshapeWindow(self.initWindowWidth,self.initWindowHeight)


if __name__ == '__main__':
    monitor_path = './tmp'
    args = get_args(monitor_path=monitor_path, model_save_path=monitor_path,
                    max_iter=20000, learning_rate=0.002, batch_size=64,
                    weight_decay=0.0001)
    #train(args)
    app = App(args)

    glutInitWindowPosition(0, 0);
    glutInitWindowSize(app.initWindowWidth, app.initWindowHeight);
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE )
    glutCreateWindow("Display")
    glutDisplayFunc(app.draw)
    glutReshapeFunc(app.reshape)
    glutKeyboardFunc(app.keyboard)
    app.init()
    glutIdleFunc(app.idle)
    glutMainLoop()
