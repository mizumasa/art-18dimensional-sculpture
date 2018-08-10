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

SIZE = 180
DRAW_RESIZE = 1.3

CAM_ON = True
CAM_ON_ONLY = [7]
CAM_SAVED = False

VIEW_ON = False
VIEW_ON = True

LOOP_MODE = True
LOOP_ON_IGNORE = [7]
LOOP_LENGTH = 50
LOOP_FADE = 40

TRAIN_ON = True
TRAIN_ON_ONLY = [7]
SAVE_ON = False

PARAM_NORM = True
LEARN_RATE = 0.0002
LEARN_RATE = 0.005

DEMO = False
if DEMO: #realtimedemo
    LEARN_RATE = 0.005
    LOOP_MODE = True
    LOOP_LENGTH = 560
    LOOP_FADE = 20

#MULTI mode
MULTI_ON = True
NET_NUM = 15
MULTI_DRAW_H = 3

def network(x, maxh=25, depth=6):
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

def net(net_num):
    return "net"+str(net_num)

class App:
    def __init__(self,args):
        self.args =args
        self.start = time.time()
        # USB camera setup
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened() is False:
            raise("IO Error")
        if CAM_ON:
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.initWindowWidth = 800 * 2
        self.initWindowHeight = 400 * 2
        self.frameStart = time.time()
        self.frameSpentTime = []
        self.canvas = np.zeros((self.initWindowHeight,self.initWindowWidth,3),dtype = "uint8")
        self.lenna = util.makeOutput("lenna.jpg",SIZE)
        self.lenna = util.makeOutput("monna.jpg",SIZE)
        self.lennaOn = False
        self.drawCenter = False
        self.param = {"pre":{},"next":{}}

        self.mParam={}
        self.mX={}
        self.mY={}
        self.mOutput={}
        self.mLoss={}
        self.mSolver={}
        self.mCountLoop = {}

        from nnabla.contrib.context import extension_context
        extension_module = args.context
        if args.context is None:
            extension_module = 'cpu'
        logger.info("Running in %s" % extension_module)
        ctx = extension_context(extension_module, device_id=args.device_id)
        nn.set_default_context(ctx)

        self.dataIn = util.makeInput(SIZE)
        self.dataOut = util.makeOutput("test.png",SIZE)
    
        if MULTI_ON:
            for idx in range(NET_NUM):
                with nn.parameter_scope(net(idx)):
                    self.mX[net(idx)] = nn.Variable([1, 3, SIZE, SIZE])
                    self.mX[net(idx)].d = self.dataIn.copy()
                    self.mY[net(idx)] = network(self.mX[net(idx)])
                    self.mOutput[net(idx)] = nn.Variable([1, 3, SIZE, SIZE])
                    self.mOutput[net(idx)].d = self.dataOut
                    self.mLoss[net(idx)] = F.mean(F.squared_error(self.mY[net(idx)], self.mOutput[net(idx)]))
                    param = nn.get_parameters()
                    for i,j in param.items():
                        param.get(i).d = np.random.randn(*(j.d.shape))
                    self.mSolver[net(idx)] = S.Adam(args.learning_rate, beta1=0.5)
                    with nn.parameter_scope("net"):
                        self.mSolver[net(idx)].set_parameters(nn.get_parameters())
                self.mInitParam(idx)
                self.mCountLoop[net(idx)] = int(np.random.rand() * (LOOP_LENGTH - 1 ) )

        else:
            self.x = nn.Variable([1, 3, SIZE, SIZE])
            #y = network(x, maxh=8, depth=5)
            self.y = network(self.x)
        
            self.output = nn.Variable([1, 3, SIZE, SIZE])
            self.output.d = self.dataOut
            self.loss = F.mean(F.squared_error(self.y, self.output))

            param = nn.get_parameters()
            for i,j in param.items():
                param.get(i).d = np.random.randn(*(j.d.shape))

            self.solver = S.Adam(args.learning_rate, beta1=0.5)
            with nn.parameter_scope("net"):
                self.solver.set_parameters(nn.get_parameters())
            self.initParam()
        
        
        self.count = 0
        self.countLoop = 0

    def draw(self):
        self.frameStart = time.time()
        # Paste into texture to draw at high speed
        if CAM_ON:
            ret, frame = self.cap.read() #read camera image
        else:
            ret, frame = (True,np.zeros((720,1280,3),dtype="uint8"))
        #img = cv2.imread('image.png') # if use the image file
        if ret:
            if CAM_ON:
                img= cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) #BGR-->RGB
                img_ = img.copy()
                contrast_converter = ImageEnhance.Contrast(Image.fromarray(img))
                img = np.asarray(contrast_converter.enhance(2.))
                if MULTI_ON:
                    for idx in range(NET_NUM):
                        if idx in CAM_ON_ONLY:
                            if self.lennaOn:
                                self.mOutput[net(idx)].d = self.lenna
                            else:
                                self.mOutput[net(idx)].d = util.makeOutputFromFrame(img,SIZE)
                            if CAM_SAVED:
                                print "use saved cam",self.tmpCam.shape
                                self.mOutput[net(idx)].d = self.tmpCam
                                print "use saved cam2",self.tmpCam.shape

                else:
                    self.output.d = util.makeOutputFromFrame(img,SIZE)
            #else:
            #    if MULTI_ON:
            #        for idx in range(NET_NUM):
            #            pass
            #            #self.mOutput[net(idx)].d = self.lenna
            #            #self.mOutput[net(idx)].d = self.dataOut
            #    else:
            #        self.output.d = self.lenna
            #        self.output.d = self.dataOut

            if LOOP_MODE:
                self.countLoop += 1
                if self.countLoop == LOOP_LENGTH:
                    self.countLoop = 0
                if MULTI_ON:
                    for idx in range(NET_NUM):
                        self.mCountLoop[net(idx)] += 1
                        if self.mCountLoop[net(idx)] == LOOP_LENGTH:
                            self.mCountLoop[net(idx)] = 0
                        if idx not in LOOP_ON_IGNORE:
                            if self.mCountLoop[net(idx)] < LOOP_FADE:
                                self.mSetNowParam(idx,1. * self.mCountLoop[net(idx)] / LOOP_FADE)
                            if self.mCountLoop[net(idx)] == LOOP_FADE:
                                self.mNextParam(idx)
                else:
                    if self.countLoop < LOOP_FADE:
                        self.setNowParam(1. * self.countLoop / LOOP_FADE)
                    if self.countLoop == LOOP_FADE:
                        self.nextParam()

            self.count += 1
            if self.count % 30 == 0 and len(self.frameSpentTime) > 0:
                print self.count, "fps:", 1. * len(self.frameSpentTime) / sum(self.frameSpentTime) 
                self.frameSpentTime = []

            if MULTI_ON:
                for idx in range(NET_NUM):
                    #self.mX[net(idx)].d = self.dataIn.copy()
                    self.mSolver[net(idx)].zero_grad()
                    self.mLoss[net(idx)].forward(clear_no_need_grad=True)
            else:
                self.x.d = self.dataIn.copy()
                self.solver.zero_grad()
                self.loss.forward(clear_no_need_grad=True)

            #cv2.imshow('screen', util.makeBGRVstack(np.concatenate([y.d, output.d], axis=2)))
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            #outframe = util.makeBGR(np.concatenate([self.y.d, util.makeOutputFromFrame(img_,SIZE)], axis=2))

            self.startCanvas()
            
            if 1:
                dSIZE = min(600,SIZE)
                if MULTI_ON:
                    for idx in range(NET_NUM):
                        marginY = (self.windowSizeH - dSIZE * DRAW_RESIZE * MULTI_DRAW_H)/2
                        marginX = (self.windowSizeW - dSIZE * DRAW_RESIZE * int(NET_NUM / MULTI_DRAW_H))/2
                        if self.drawCenter:
                            marginX = 0;marginY=0
                        self.drawImage(util.makeBGR(self.mY[net(idx)].d),dSIZE*(int(idx/MULTI_DRAW_H))*DRAW_RESIZE,dSIZE*(idx%MULTI_DRAW_H)*DRAW_RESIZE,dSIZE*DRAW_RESIZE,dSIZE*DRAW_RESIZE,mx = marginX,my=marginY)
                        #self.drawImage(util.makeBGR(self.mOutput[net(idx)].d),dSIZE*(int(idx/MULTI_DRAW_H)),dSIZE*(idx%MULTI_DRAW_H),dSIZE,dSIZE)
                else:
                    self.drawImage(util.makeBGR(self.y.d),dSIZE,0,dSIZE,dSIZE)
                    self.drawImage(util.makeBGR(self.output.d),0,0,dSIZE,dSIZE)
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
                if MULTI_ON:
                    data = glReadPixels(0, 0, self.windowSizeW, self.windowSizeH, GL_RGBA, GL_UNSIGNED_BYTE)
                    image = Image.frombytes("RGBA", (self.windowSizeW, self.windowSizeH), data)
                    image.save(os.path.join(self.args.model_save_path, "output_%06d.png" % self.count))
                else:
                    img2 = util.makePng(self.y.d)
                    img2.save(os.path.join(self.args.model_save_path, "output_%06d.png" % self.count))
            if TRAIN_ON:
                if MULTI_ON:
                    for idx in range(NET_NUM):
                        if idx in TRAIN_ON_ONLY:
                            self.mLoss[net(idx)].backward(clear_buffer=True)
                            self.mSolver[net(idx)].weight_decay(self.args.weight_decay)
                            self.mSolver[net(idx)].update()
                else:
                    self.loss.backward(clear_buffer=True)
                    self.solver.weight_decay(self.args.weight_decay)
                    self.solver.update()
                if PARAM_NORM:
                    param = nn.get_parameters()
                    for i,j in param.items():
                        param.get(i).d /= param.get(i).d.std() 
        self.frameSpentTime.append(time.time() - self.frameStart)
        return

    def drawImage(self,ary,x,y,w,h,useCanvas = True,mx = 0,my = 0):
        if useCanvas:
            if (ary.shape == (h,w,3)):
                try:
                    self.canvas[y+my:y+my+h,x+mx:x+mx+w,:] = ary
                except:
                    print "WARN out of area",(x,y,w,h)
            else:
                try:
                    self.canvas[int(y+my):int(y+my+h),int(x+mx):int(x+mx+w),:] = cv2.resize(ary,(int(w),int(h)))
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
            buf = self.param["next"][i] * level + self.param["pre"][i] * (1. - level)
            param.get(i).d = (buf / buf.std()) 
        return

    def mInitParam(self,idx):
        with nn.parameter_scope(net(idx)):
            param = nn.get_parameters()
            self.mParam[net(idx)] = {"pre":{},"next":{}}
            for i,j in param.items():
                self.mParam[net(idx)]["pre"][i] = np.random.randn(*(j.d.shape))
                self.mParam[net(idx)]["next"][i] = np.random.randn(*(j.d.shape))
        return

    def mNextParam(self,idx):
        if np.random.rand() > 0.8:
            with nn.parameter_scope(net(7)):
                param = nn.get_parameters()
                for i,j in param.items():
                    self.mParam[net(idx)]["pre"][i] = self.mParam[net(idx)]["next"][i].copy()
                    self.mParam[net(idx)]["next"][i] = param.get(i).d
        else:
            for i in self.mParam[net(idx)]["next"].keys():
                self.mParam[net(idx)]["pre"][i] = self.mParam[net(idx)]["next"][i].copy()
                self.mParam[net(idx)]["next"][i] = np.random.randn(*(self.mParam[net(idx)]["next"][i].shape))
            return
        return

    def mSetNowParam(self,idx,level):#level = 0 -> 1
        with nn.parameter_scope(net(idx)):
            param = nn.get_parameters()
            for i,j in param.items():
                buf = self.mParam[net(idx)]["next"][i] * level + self.mParam[net(idx)]["pre"][i] * (1. - level)
                param.get(i).d = (buf / buf.std()) 
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
        #glutFullScreen()
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
        if key == 'c':
            print("center")
            self.drawCenter = not self.drawCenter
        if key == 'q':
            print('exit')
            sys.exit()
        if key == 'n':
            #self.setNextParam()
            self.nextParam()
            self.setNowParam(0.5)
        if key == 'l':
            self.lennaOn = not self.lennaOn
        if key == 't':
            global TRAIN_ON
            TRAIN_ON = not TRAIN_ON
        if key == 's':
            global SAVE_ON
            SAVE_ON = not SAVE_ON
        if key == 'y':
            self.tmpCam = self.mOutput[net(7)].d.copy()
        if key == 'u':
            global CAM_SAVED
            CAM_SAVED = not CAM_SAVED
        if key == 'f':
            if self.initWindowWidth == self.windowSizeW:
                glutFullScreen()
            else:
                glutReshapeWindow(self.initWindowWidth,self.initWindowHeight)


if __name__ == '__main__':
    monitor_path = './tmpLoop11'
    args = get_args(monitor_path=monitor_path, model_save_path=monitor_path,
                    max_iter=20000, learning_rate=LEARN_RATE, batch_size=64,
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
