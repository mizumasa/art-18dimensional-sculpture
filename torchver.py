# -*- coding: utf-8 -*-
from __future__ import absolute_import
from six.moves import range

import os
import sys
import time
import glob
import numpy as np
from PIL import Image, ImageEnhance

import cv2

# PyTorch 関連
import torch
import torch.nn as nn
import torch.optim as optim

# OpenGL 関連
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

# (nnabla の args.py 相当と同じインターフェースで引数をパースすると仮定)
from args import get_args


# ============ ここから：NNablaのネットワーク部分をPyTorchへ変換 ============

SIZE = 180
DRAW_RESIZE = 1.3

CAM_ON = True
CAM_ON_ONLY = [7]
CAM_SAVED = False

VIEW_ON = True
LOOP_MODE = True
LOOP_ON_IGNORE = [7]
LOOP_LENGTH = 50
LOOP_FADE = 40
TRAIN_ON = True
TRAIN_ON_ONLY = [7]
SAVE_ON = False

PARAM_NORM = True       # パラメータ正規化
LEARN_RATE = 0.005
DEMO = False
if DEMO:  # realtimedemo
    LEARN_RATE = 0.005
    LOOP_MODE = True
    LOOP_LENGTH = 560
    LOOP_FADE = 20

MULTI_ON = True
NET_NUM = 15
MULTI_DRAW_H = 3

#--------------------------------------------------------------------------
# PyTorch版ネットワーク定義
# nnabla の network(x) に相当
# (1, 3, H, W) → conv(1x1, maxh channel) → Tanh → ... → conv(1x1, 3 channel) → Sigmoid
#--------------------------------------------------------------------------
class SimpleNet(nn.Module):
    def __init__(self, maxh=25, depth=6):
        super(SimpleNet, self).__init__()
        layers = []
        # 入力: 3ch → maxh ch (1x1 conv)
        layers.append(nn.Conv2d(3, maxh, kernel_size=1, stride=1, bias=False))
        layers.append(nn.Tanh())

        # depth 回の(1x1 conv → Tanh)
        for _ in range(depth):
            layers.append(nn.Conv2d(maxh, maxh, kernel_size=1, stride=1, bias=False))
            layers.append(nn.Tanh())

        # 出力: maxh ch → 3ch (1x1 conv)
        self.conv_out = nn.Conv2d(maxh, 3, kernel_size=1, stride=1, bias=False)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        x = self.conv_out(x)
        # 最終活性は Sigmoid
        x = torch.sigmoid(x)
        return x

#--------------------------------------------------------------------------
# 各種画像処理系関数 (nnablaのまま/PyTorch化と直接関係なし)
#--------------------------------------------------------------------------
def makeOutput(filename, size):
    img = Image.open(filename)
    # コントラスト上げる例
    contrast_converter = ImageEnhance.Contrast(img)
    img = contrast_converter.enhance(2.0)

    if img.size != (size, size):
        print("resize")
        img = img.resize((size, size))
    imgary = np.asarray(img).copy()

    # BGRなどをいじるロジックがあれば適宜
    ary = np.zeros((1, 3, size, size), dtype=np.float32)
    for i in range(3):
        ary[0][i] = imgary[:, :, i] / 255.0
    return ary

def makeOutputFromFrame(frame, size):
    imgary = cv2.resize(frame, (size, size))
    ary = np.zeros((1, 3, size, size), dtype=np.float32)
    for i in range(3):
        ary[0][i] = imgary[:, :, i] / 255.0
    return ary

def makeInput(size):
    # もともとデモ用にRGBの各チャンネル等を初期化
    ary = np.zeros((1, 3, size, size), dtype=np.float32)
    ary[0][2][:,:] = 1.0
    for i in range(size):
        ary[0][0][i, :] = i * 1.0 / size
        ary[0][1][:, i] = i * 1.0 / size
    return ary

def makeBGR(tensor_np):
    """ (1,3,H,W) → (H,W,3) のuint8画像に変換 """
    h = tensor_np.shape[2]
    w = tensor_np.shape[3]
    output = np.zeros((h, w, 3), dtype="uint8")
    for i in range(3):
        output[:, :, i] = np.uint8(tensor_np[0, i, :, :] * 255)
    return output

def makePng(tensor_np):
    """ (1,3,H,W) → PIL.Image """
    return Image.fromarray(makeBGR(tensor_np))

#--------------------------------------------------------------------------
# メインアプリケーションクラス (OpenGL描画・カメラ入力・学習ループ)
#--------------------------------------------------------------------------
class App:
    def __init__(self, args):
        self.args = args
        self.start = time.time()

        # USB camera setup
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera.")
        if CAM_ON:
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.initWindowWidth = 800 * 2
        self.initWindowHeight = 400 * 2
        self.frameStart = time.time()
        self.frameSpentTime = []

        self.canvas = np.zeros((self.initWindowHeight, self.initWindowWidth, 3), dtype="uint8")
        self.lenna = makeOutput("lenna.jpg", SIZE)
        self.lenna = makeOutput("monna.jpg", SIZE)
        self.lennaOn = False
        self.drawCenter = False

        # PyTorch 用デバイス設定 (GPUがあれば使う)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 損失関数 (nnabla の F.mean(F.squared_error) 相当)
        self.criterion = nn.MSELoss(reduction='mean')

        # ここでは引数に対する weight_decay を optimizer で設定するため保存
        self.weight_decay = args.weight_decay
        self.learning_rate = args.learning_rate

        # 入出力用 (numpy) バッファ
        self.dataIn = makeInput(SIZE)  # (1,3,SIZE,SIZE)
        self.dataOut = makeOutput("test.png", SIZE)

        #----------------------------------------------------------------------
        # MULTI_ON の場合は、NET_NUM 個のモデル/optimizer/param管理をそれぞれ持つ
        #----------------------------------------------------------------------
        self.mModels = {}
        self.mOptimizers = {}
        self.mX = {}
        self.mOutput = {}
        self.mLoss = {}
        self.mCountLoop = {}
        self.mParam = {}

        if MULTI_ON:
            for idx in range(NET_NUM):
                # モデル作成
                model = SimpleNet(maxh=25, depth=6).to(self.device)
                # Optimizer (nnablaのAdam(learning_rate, beta1=0.5)相当)
                optimizer = optim.Adam(model.parameters(),
                                       lr=self.learning_rate,
                                       betas=(0.5, 0.999),
                                       weight_decay=self.weight_decay)

                self.mModels[idx] = model
                self.mOptimizers[idx] = optimizer

                # 「入力と出力」の tensor を確保 (1,3,SIZE,SIZE)
                # 実際の画像などは都度 numpy→tensor でコピー
                self.mX[idx] = torch.zeros((1, 3, SIZE, SIZE), dtype=torch.float32, device=self.device)
                self.mOutput[idx] = torch.zeros((1, 3, SIZE, SIZE), dtype=torch.float32, device=self.device)

                # 損失を後で計算するので予約
                self.mLoss[idx] = None

                # パラメータの初期値をランダムに
                with torch.no_grad():
                    for p in model.parameters():
                        p.data.normal_()

                # nextParam / preParam 用の管理テーブルを作る
                self.mParam[idx] = {
                    "pre": {},
                    "next": {},
                }
                self.mInitParam(idx)  # pre/next にランダムな値を仕込む

                # ループカウンタ
                self.mCountLoop[idx] = int(np.random.rand() * (LOOP_LENGTH - 1))

        else:
            # シングルネットワーク版 (参考: nnabla版)
            self.model = SimpleNet(maxh=25, depth=6).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=self.learning_rate,
                                        betas=(0.5, 0.999),
                                        weight_decay=self.weight_decay)

            # 入力出力のテンソル
            self.x = torch.zeros((1, 3, SIZE, SIZE), dtype=torch.float32, device=self.device)
            self.output = torch.zeros((1, 3, SIZE, SIZE), dtype=torch.float32, device=self.device)

            # パラメータ初期化
            with torch.no_grad():
                for p in self.model.parameters():
                    p.data.normal_()

            self.param = {"pre": {}, "next": {}}
            self.initParam()

        self.count = 0
        self.countLoop = 0

    #----------------------------------------------------------------------
    # mainの描画関数 (glutDisplayFunc から呼ばれる)
    #----------------------------------------------------------------------
    def draw(self):
        self.frameStart = time.time()

        # カメラ画像取得
        if CAM_ON:
            ret, frame = self.cap.read()
        else:
            ret, frame = True, np.zeros((720, 1280, 3), dtype="uint8")

        if ret:
            if CAM_ON:
                # BGR -> RGB
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # コントラスト強調してみる (nnabla版にあわせる)
                contrast_converter = ImageEnhance.Contrast(Image.fromarray(img))
                img = np.asarray(contrast_converter.enhance(2.0))

                if MULTI_ON:
                    for idx in range(NET_NUM):
                        if idx in CAM_ON_ONLY:
                            if self.lennaOn:
                                self.mOutput[idx].data = torch.from_numpy(self.lenna).to(self.device)
                            else:
                                self.mOutput[idx].data = torch.from_numpy(
                                    makeOutputFromFrame(img, SIZE)
                                ).to(self.device)
                            if CAM_SAVED:
                                # self.tmpCam を保存している場合の例
                                self.mOutput[idx].data = self.tmpCam.clone().to(self.device)
                else:
                    self.output.data = torch.from_numpy(
                        makeOutputFromFrame(img, SIZE)
                    ).to(self.device)

            # ループでパラメータを補間するモード
            if LOOP_MODE:
                self.countLoop += 1
                if self.countLoop == LOOP_LENGTH:
                    self.countLoop = 0

                if MULTI_ON:
                    for idx in range(NET_NUM):
                        self.mCountLoop[idx] += 1
                        if self.mCountLoop[idx] == LOOP_LENGTH:
                            self.mCountLoop[idx] = 0
                        if idx not in LOOP_ON_IGNORE:
                            # LOOP_FADE フレームかけて pre→next に補間
                            if self.mCountLoop[idx] < LOOP_FADE:
                                level = float(self.mCountLoop[idx]) / LOOP_FADE
                                self.mSetNowParam(idx, level)
                            if self.mCountLoop[idx] == LOOP_FADE:
                                self.mNextParam(idx)
                else:
                    if self.countLoop < LOOP_FADE:
                        level = float(self.countLoop) / LOOP_FADE
                        self.setNowParam(level)
                    if self.countLoop == LOOP_FADE:
                        self.nextParam()

            self.count += 1
            if (self.count % 30) == 0 and len(self.frameSpentTime) > 0:
                fps = 1.0 * len(self.frameSpentTime) / sum(self.frameSpentTime)
                print(self.count, "fps:", fps)
                self.frameSpentTime = []

            # 学習の順番:
            # 1. 入力 (x, or mX[idx]) に self.dataIn をセット
            # 2. 順伝播
            # 3. 損失計算
            # 4. backward() + optimizer.step()

            if MULTI_ON:
                # forward
                for idx in range(NET_NUM):
                    # とりあえず x を self.dataIn として固定
                    self.mX[idx].data = torch.from_numpy(self.dataIn).to(self.device)

                    # 学習しないなら skip
                    if TRAIN_ON and (idx in TRAIN_ON_ONLY):
                        self.mOptimizers[idx].zero_grad()
                        y_pred = self.mModels[idx](self.mX[idx])
                        loss = self.criterion(y_pred, self.mOutput[idx])
                        self.mLoss[idx] = loss

                    else:
                        # 学習しないならフォワード計算のみ
                        with torch.no_grad():
                            y_pred = self.mModels[idx](self.mX[idx])
                        self.mLoss[idx] = None

            else:
                # シングルネットワーク版
                self.x.data = torch.from_numpy(self.dataIn).to(self.device)
                self.optimizer.zero_grad()
                y_pred = self.model(self.x)
                loss = self.criterion(y_pred, self.output)
                self.loss = loss

            # 描画フェーズ (OpenGL)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.startCanvas()

            # 画面に (推論結果, 入力 or カメラ画像) を並べて描画
            dSIZE = min(600, SIZE)
            if MULTI_ON:
                # mModels[idx] の推論結果 y_pred は学習時に保持していないため
                # 今回のサンプルでは再度 forward して結果を描画用に取得
                # (性能が必要なら、学習時に保存しておくのがよい)
                for idx in range(NET_NUM):
                    with torch.no_grad():
                        tmp_pred = self.mModels[idx](self.mX[idx])
                    out_img = tmp_pred.detach().cpu().numpy()
                    # 画像をcanvasへ
                    marginY = (self.windowSizeH - dSIZE * DRAW_RESIZE * MULTI_DRAW_H) / 2
                    marginX = (self.windowSizeW - dSIZE * DRAW_RESIZE * (NET_NUM // MULTI_DRAW_H)) / 2
                    if self.drawCenter:
                        marginX, marginY = 0, 0
                    self.drawImage(
                        makeBGR(out_img),
                        dSIZE * (idx // MULTI_DRAW_H) * DRAW_RESIZE,
                        dSIZE * (idx % MULTI_DRAW_H) * DRAW_RESIZE,
                        dSIZE * DRAW_RESIZE,
                        dSIZE * DRAW_RESIZE,
                        useCanvas=True,
                        mx=marginX,
                        my=marginY
                    )
            else:
                # シングル版は一度 forward した y_pred があるのでそれを使う
                y_img = y_pred.detach().cpu().numpy()
                self.drawImage(makeBGR(y_img), dSIZE, 0, dSIZE, dSIZE)
                out_img = self.output.detach().cpu().numpy()
                self.drawImage(makeBGR(out_img), 0, 0, dSIZE, dSIZE)

            self.drawCanvas()
            glFlush()
            glutSwapBuffers()

            # 保存フラグが立っていれば画像を保存
            if SAVE_ON and (self.count % 1 == 0):
                # 全画面OpenGLピクセルを読む
                data = glReadPixels(0, 0, self.windowSizeW, self.windowSizeH, GL_RGBA, GL_UNSIGNED_BYTE)
                image = Image.frombytes("RGBA", (self.windowSizeW, self.windowSizeH), data)
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                image.save(os.path.join(self.args.model_save_path, "output_%06d.png" % self.count))

            # 学習実行
            if TRAIN_ON:
                if MULTI_ON:
                    for idx in range(NET_NUM):
                        if idx in TRAIN_ON_ONLY and self.mLoss[idx] is not None:
                            self.mLoss[idx].backward()
                            self.mOptimizers[idx].step()

                    # パラメータ正規化
                    if PARAM_NORM:
                        for idx in range(NET_NUM):
                            if idx in TRAIN_ON_ONLY:
                                with torch.no_grad():
                                    for p in self.mModels[idx].parameters():
                                        std_val = p.std().item()
                                        if std_val > 1e-9:
                                            p.div_(std_val)

                else:
                    self.loss.backward()
                    self.optimizer.step()

                    if PARAM_NORM:
                        with torch.no_grad():
                            for p in self.model.parameters():
                                std_val = p.std().item()
                                if std_val > 1e-9:
                                    p.div_(std_val)

        self.frameSpentTime.append(time.time() - self.frameStart)

    #--------------------------------------------------------------------------
    # 画像を self.canvas に貼る / あるいはテクスチャとして描画
    #--------------------------------------------------------------------------
    def drawImage(self, ary, x, y, w, h, useCanvas=True, mx=0, my=0):
        """ary: (H_,W_,3) のnumpy画像(BGR)"""
        if useCanvas:
            # canvas上に貼り付け
            # サイズが合わない場合はリサイズ
            H_, W_ = ary.shape[:2]
            x0, y0 = int(x + mx), int(y + my)
            x1, y1 = x0 + int(w), y0 + int(h)
            try:
                if (H_, W_) == (int(h), int(w)):
                    self.canvas[y0:y1, x0:x1, :] = ary
                else:
                    self.canvas[y0:y1, x0:x1, :] = cv2.resize(ary, (x1 - x0, y1 - y0))
            except:
                # 描画範囲がはみ出す可能性あり
                pass
            return

        # OpenGLテクスチャとして貼り付けたい場合の例
        h_, w_ = ary.shape[:2]
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w_, h_, 0, GL_RGB, GL_UNSIGNED_BYTE, ary)
        glColor3f(1.0, 1.0, 1.0)
        glEnable(GL_TEXTURE_2D)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        left = 2.0 * x / self.windowSizeW - 1.0
        top = 2.0 * y / self.windowSizeH - 1.0
        right = 2.0 * (x + w) / self.windowSizeW - 1.0
        bottom = 2.0 * (y + h) / self.windowSizeH - 1.0

        glBegin(GL_QUADS)
        glTexCoord2d(0.0, 1.0)
        glVertex3d(left, -bottom, 0.0)
        glTexCoord2d(1.0, 1.0)
        glVertex3d(right, -bottom, 0.0)
        glTexCoord2d(1.0, 0.0)
        glVertex3d(right, -top, 0.0)
        glTexCoord2d(0.0, 0.0)
        glVertex3d(left, -top, 0.0)
        glEnd()

    #--------------------------------------------------------------------------
    # シングルネットワーク用パラメータ管理 (nnabla版 nextParam/setNowParam)
    #--------------------------------------------------------------------------
    def initParam(self):
        # pre / next に乱数を生成しておく
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                self.param["pre"][name] = torch.randn_like(p)
                self.param["next"][name] = torch.randn_like(p)

    def nextParam(self):
        # pre <- next をコピーしてから next に新しい乱数を仕込む
        with torch.no_grad():
            for name, _ in self.model.named_parameters():
                self.param["pre"][name] = self.param["next"][name].clone()
                self.param["next"][name] = torch.randn_like(self.param["next"][name])

    def setNowParam(self, level):
        # 現在のモデルパラメータ = pre*(1-level) + next*level を正規化
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                buf = self.param["pre"][name] * (1.0 - level) + self.param["next"][name] * level
                std_val = buf.std().item()
                if std_val > 1e-9:
                    buf = buf / std_val
                p.data.copy_(buf)

    #--------------------------------------------------------------------------
    # マルチネットワーク用パラメータ管理
    #--------------------------------------------------------------------------
    def mInitParam(self, idx):
        # net(idx) の pre / next を乱数に
        with torch.no_grad():
            for name, p in self.mModels[idx].named_parameters():
                self.mParam[idx]["pre"][name] = torch.randn_like(p)
                self.mParam[idx]["next"][name] = torch.randn_like(p)

    def mNextParam(self, idx):
        # ランダムに 80% の確率で通常の new random,
        # 20% の確率で net(7) の現在パラメータをコピー (nnabla版に準拠)
        with torch.no_grad():
            if np.random.rand() > 0.8:
                # net(7) の現パラメータをコピー
                for name, p7 in self.mModels[7].named_parameters():
                    self.mParam[idx]["pre"][name] = self.mParam[idx]["next"][name].clone()
                    self.mParam[idx]["next"][name] = p7.clone()
            else:
                # 通常ランダム
                for name in self.mParam[idx]["next"].keys():
                    self.mParam[idx]["pre"][name] = self.mParam[idx]["next"][name].clone()
                    self.mParam[idx]["next"][name] = torch.randn_like(self.mParam[idx]["next"][name])

    def mSetNowParam(self, idx, level):
        # param = pre*(1-level) + next*level を正規化
        with torch.no_grad():
            for name, p in self.mModels[idx].named_parameters():
                buf = self.mParam[idx]["pre"][name] * (1.0 - level) + self.mParam[idx]["next"][name] * level
                std_val = buf.std().item()
                if std_val > 1e-9:
                    buf = buf / std_val
                p.data.copy_(buf)

    #--------------------------------------------------------------------------
    # OpenGL描画まわりの処理
    #--------------------------------------------------------------------------
    def startCanvas(self):
        if self.canvas.shape != (self.windowSizeH, self.windowSizeW, 3):
            self.canvas = np.zeros((self.windowSizeH, self.windowSizeW, 3), dtype="uint8")
        else:
            self.canvas[:,:,:] = 0

    def drawCanvas(self):
        # self.canvas をテクスチャとして描画
        self.drawImage(self.canvas, 0, 0, self.windowSizeW, self.windowSizeH, useCanvas=False)

    def init(self):
        glClearColor(0.7, 0.7, 0.7, 0.7)

    def idle(self):
        glutPostRedisplay()

    def reshape(self, w, h):
        self.windowSizeW = w
        self.windowSizeH = h
        glViewport(0, 0, w, h)
        glLoadIdentity()

    def keyboard(self, key, x, y):
        key = key.decode('utf-8')
        global TRAIN_ON, SAVE_ON, CAM_SAVED
        if key == 'c':
            self.drawCenter = not self.drawCenter
        if key == 'q':
            print('exit')
            sys.exit()
        if key == 'n':
            self.nextParam()
            self.setNowParam(0.5)
        if key == 'l':
            self.lennaOn = not self.lennaOn
        if key == 't':
            TRAIN_ON = not TRAIN_ON
        if key == 's':
            SAVE_ON = not SAVE_ON
        if key == 'y':
            # net(7) の出力を一時保存
            self.tmpCam = self.mOutput[7].clone()
        if key == 'u':
            CAM_SAVED = not CAM_SAVED
        if key == 'f':
            if self.initWindowWidth == self.windowSizeW:
                glutFullScreen()
            else:
                glutReshapeWindow(self.initWindowWidth, self.initWindowHeight)


#--------------------------------------------------------------------------
# メインエントリ
#--------------------------------------------------------------------------
if __name__ == '__main__':
    monitor_path = './tmpLoop11'
    args = get_args(monitor_path=monitor_path,
                    model_save_path=monitor_path,
                    max_iter=20000,
                    learning_rate=LEARN_RATE,
                    batch_size=64,
                    weight_decay=0.0001)

    app = App(args)
    glutInitWindowPosition(0, 0)
    glutInitWindowSize(app.initWindowWidth, app.initWindowHeight)
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE)
    glutCreateWindow(b'Display')
    glutDisplayFunc(app.draw)
    glutReshapeFunc(app.reshape)
    glutKeyboardFunc(app.keyboard)
    app.init()
    glutIdleFunc(app.idle)
    glutMainLoop()
