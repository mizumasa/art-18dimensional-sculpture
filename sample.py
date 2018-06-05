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
#from mnist_data import data_iterator_mnist

import os

import iterator

VEC_SIZE = 500
SIZE = 320

def load_kanji_data():
	#imgs = glob.glob("./fontImgUtsukushi/*.png")
    imgs = glob.glob("./fontImg56/*.png")
    print len(imgs)
    imgs += glob.glob("./fontImgH56/*.png")
    print len(imgs)
    value = np.zeros((len(imgs),1,56,56))
    label = np.zeros((len(imgs),100))
    for i,j in enumerate(imgs):
        value[i][0]=np.asarray(Image.open(j))
        print i
    return (value,label)

def generator(z, maxh=256, test=False, output_hidden=False):
    """
    Building generator network which takes (B, Z, 1, 1) inputs and generates
    (B, 1, 28, 28) outputs.
    """
    # Define shortcut functions
    def bn(x):
        # Batch normalization
        return PF.batch_normalization(x, batch_stat=not test)

    def upsample2(x, c):
        # Twise upsampling with deconvolution.
        return PF.deconvolution(x, c, kernel=(4, 4), pad=(1, 1), stride=(2, 2), with_bias=False)

    assert maxh / 4 > 0
    with nn.parameter_scope("gen"):
        # (Z, 1, 1) --> (256, 4, 4)
        with nn.parameter_scope("deconv1"):
            d1 = F.elu(bn(PF.deconvolution(z, maxh, (4, 4), with_bias=False)))
        # (256, 4, 4) --> (128, 8, 8)
        with nn.parameter_scope("deconv2"):
            d2 = F.elu(bn(upsample2(d1, maxh / 2)))
        # (128, 8, 8) --> (64, 16, 16)
        with nn.parameter_scope("deconv3"):
            d3 = F.elu(bn(upsample2(d2, maxh / 4)))
        # (64, 16, 16) --> (32, 28, 28)
        with nn.parameter_scope("deconv4"):
            # Convolution with kernel=4, pad=3 and stride=2 transforms a 28 x 28 map
            # to a 16 x 16 map. Deconvolution with those parameters behaves like an
            # inverse operation, i.e. maps 16 x 16 to 28 x 28.
            d4 = F.elu(bn(PF.deconvolution(
                d3, maxh / 8, (4, 4), pad=(3, 3), stride=(2, 2), with_bias=False)))
        # (32, 28, 28) --> (32, 56, 56)
        with nn.parameter_scope("deconv5"):
            # Convolution with kernel=4, pad=3 and stride=2 transforms a 28 x 28 map
            # to a 16 x 16 map. Deconvolution with those parameters behaves like an
            # inverse operation, i.e. maps 16 x 16 to 28 x 28.
            d5 = F.elu(bn(upsample2(d4, maxh / 8)))
        # (32, 56, 56) --> (1, 56, 56)
        with nn.parameter_scope("conv6"):
            x = F.tanh(PF.convolution(d5, 1, (3, 3), pad=(1, 1)))
    if output_hidden:
        return x, [d1, d2, d3, d4]
    return x


def discriminator(x, maxh=256, test=False, output_hidden=False):
    """
    Building discriminator network which maps a (B, 1, 28, 28) input to
    a (B, 1).
    """
    # Define shortcut functions
    def bn(xx):
        # Batch normalization
        return PF.batch_normalization(xx, batch_stat=not test)

    def downsample2(xx, c):
        return PF.convolution(xx, c, (3, 3), pad=(1, 1), stride=(2, 2), with_bias=False)

    assert maxh / 8 > 0
    with nn.parameter_scope("dis"):
        # (1, 56, 56) --> (32, 28, 28)
        with nn.parameter_scope("conv0"):
            c0 = F.elu(bn(downsample2(x, maxh / 8)))
        if not test:
            c0 = F.dropout(c0, 0.2)
        # (32, 28, 28) --> (32, 16, 16)
        with nn.parameter_scope("conv1"):
            c1 = F.elu(bn(PF.convolution(c0, maxh / 8,
                                         (3, 3), pad=(3, 3), stride=(2, 2), with_bias=False)))
        if not test:
            c1 = F.dropout(c1, 0.2)
        # (32, 16, 16) --> (64, 8, 8)
        with nn.parameter_scope("conv2"):
            c2 = F.elu(bn(downsample2(c1, maxh / 4)))
        # (64, 8, 8) --> (128, 4, 4)
        with nn.parameter_scope("conv3"):
            c3 = F.elu(bn(downsample2(c2, maxh / 2)))
        # (128, 4, 4) --> (256, 4, 4)
        with nn.parameter_scope("conv4"):
            c4 = bn(PF.convolution(c3, maxh, (3, 3),
                                   pad=(1, 1), with_bias=False))
        # (256, 4, 4) --> (1,)
        with nn.parameter_scope("fc1"):
            f = PF.affine(c4, 1)
    if output_hidden:
        return f, [c1, c2, c3, c4]
    return f

def vectorizer(x, maxh=256, test=False, output_hidden=False):
    """
    Building discriminator network which maps a (B, 1, 28, 28) input to
    a (B, 100).
    """
    # Define shortcut functions
    def bn(xx):
        # Batch normalization
        return PF.batch_normalization(xx, batch_stat=not test)

    def downsample2(xx, c):
        return PF.convolution(xx, c, (3, 3), pad=(1, 1), stride=(2, 2), with_bias=False)

    assert maxh / 8 > 0
    with nn.parameter_scope("vec"):
        # (1, 28, 28) --> (32, 16, 16)
        if not test:
            x_ = F.image_augmentation(x,min_scale=0.9,max_scale=1.08)
            x2 = F.random_shift(x_,(2,2))
            # (1, 56, 56) --> (32, 28, 28)
            with nn.parameter_scope("conv0"):
                c0 = F.elu(bn(downsample2(x2, maxh / 8)))
            with nn.parameter_scope("conv1"):
                c1 = F.elu(bn(PF.convolution(c0, maxh / 8,
                                             (3, 3), pad=(3, 3), stride=(2, 2), with_bias=False)))
        else:
            # (1, 56, 56) --> (32, 28, 28)
            with nn.parameter_scope("conv0"):
                c0 = F.elu(bn(downsample2(x, maxh / 8)))
            with nn.parameter_scope("conv1"):
                c1 = F.elu(bn(PF.convolution(c0, maxh / 8,
                                             (3, 3), pad=(3, 3), stride=(2, 2), with_bias=False)))
        # (32, 16, 16) --> (64, 8, 8)
        with nn.parameter_scope("conv2"):
            c2 = F.elu(bn(downsample2(c1, maxh / 4)))
        # (64, 8, 8) --> (128, 4, 4)
        with nn.parameter_scope("conv3"):
            c3 = F.elu(bn(downsample2(c2, maxh / 2)))
        # (128, 4, 4) --> (256, 4, 4)
        with nn.parameter_scope("conv4"):
            c4 = bn(PF.convolution(c3, maxh, (3, 3),
                                   pad=(1, 1), with_bias=False))
        # (256, 4, 4) --> (1,)
        with nn.parameter_scope("fc1"):
            #print "c4fdafafa",c4.shape
            #f = PF.affine(c4, 100)
            f = bn(PF.convolution(c4, VEC_SIZE, (4, 4),
                                   pad=(0, 0), with_bias=False))
    if output_hidden:
        return f, [c1, c2, c3, c4]
    return f



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

def makeInput():
    ary = np.zeros((1,3,SIZE,SIZE))
    ary[0][2][:,:] = 1.0
    for i in range(SIZE):
        ary[0][0][i,:] = i * 1. / SIZE
        ary[0][1][:,i] = i * 1. / SIZE
    return ary

def makePng(ary):
    #print "a max",max(ary.flatten()),"a min",min(ary.flatten())
    output = np.zeros((SIZE,SIZE,3),dtype="uint8")
    for i in range(3):
        output[:,:,i] = np.uint8(ary[0,i,:,:]*255)
    #print "o max",max(output.flatten()),"o min",min(output.flatten())
    return Image.fromarray(output)

def train(args):
    from nnabla.contrib.context import extension_context
    extension_module = args.context
    if args.context is None:
        extension_module = 'cpu'
    logger.info("Running in %s" % extension_module)
    ctx = extension_context(extension_module, device_id=args.device_id)
    nn.set_default_context(ctx)


    x = nn.Variable([1, 3, SIZE, SIZE])
    y = network(x)
    dataIn = makeInput()

    x.d = dataIn.copy()
    y.forward()
    img = makePng(y.d)
    img.save(os.path.join(args.model_save_path, "first.png"))

    output = nn.Variable([1, 3, SIZE, SIZE])
    dataOut = makeOutput("test.png")
    output.d = dataOut

    #loss = F.mean(F.sigmoid_cross_entropy(y, output))
    loss = F.mean(F.squared_error(y, output))

    param = nn.get_parameters()
    for i,j in param.items():
        param.get(i).d = np.random.randn(*(j.d.shape))

    solver = S.Adam(args.learning_rate, beta1=0.5)
    with nn.parameter_scope("net"):
        solver.set_parameters(nn.get_parameters())

    # Create monitor.
    import nnabla.monitor as M
    monitor = M.Monitor(args.monitor_path)
    monitor_loss_gen = M.MonitorSeries("Generator loss", monitor, interval=10)
    monitor_time = M.MonitorTimeElapsed("Time", monitor, interval=100)
    monitor_gen = M.MonitorImageTile(
        "gen images", monitor)

    #data = data_iterator_mnist(args.batch_size, True)
    with nn.parameter_scope("net"):
        param = nn.get_parameters()
        print param.get("conv0/conv/W").d.reshape((16,16))[:10,:10]

    # Training loop.
    for i in range(args.max_iter):
        if i % args.model_save_interval == 0:
            with nn.parameter_scope("net"):
                nn.save_parameters(os.path.join(
                    args.model_save_path, "generator_param_%06d.h5" % i))

        # Training forward
        x.d = dataIn.copy()
        # Generator update.
        solver.zero_grad()
        loss.forward(clear_no_need_grad=True)
        if i % 10 == 0:
            img = makePng(y.d)
            img.save(os.path.join(args.model_save_path, "output_%06d.png" % i))
        #print "max",max(y.d.flatten()),"min",min(y.d.flatten())
        loss.backward(clear_buffer=True)
        solver.weight_decay(args.weight_decay)
        solver.update()

        monitor_gen.add(i, y)
        monitor_loss_gen.add(i, loss.d.copy())
        monitor_time.add(i)


    with nn.parameter_scope("net"):
        nn.save_parameters(os.path.join(
            args.model_save_path, "generator_param_%06d.h5" % i))

    return


def train2(args):
    """
    Main script.
    """

    # Get context.
    from nnabla.contrib.context import extension_context
    extension_module = args.context
    if args.context is None:
        extension_module = 'cpu'
    logger.info("Running in %s" % extension_module)
    ctx = extension_context(extension_module, device_id=args.device_id)
    nn.set_default_context(ctx)

    # Create CNN network for both training and testing.
    # TRAIN

    # Fake path
    x1 = nn.Variable([args.batch_size, 1, 56, 56])
    
    #z = nn.Variable([args.batch_size, VEC_SIZE, 1, 1])
    #z = vectorizer(x1,maxh = 1024)
    #fake = generator(z,maxh= 1024)
    z_vec = vectorizer(x1)
    z = z_vec.unlinked()
    #fake2 = generator(z_vec,maxh=512)
    #fake = generator(z,maxh=512)
    fake2 = generator(z_vec)
    fake = generator(z)
    fake.persistent = True  # Not to clear at backward
    pred_fake = discriminator(fake)
    loss_gen = F.mean(F.sigmoid_cross_entropy(
        pred_fake, F.constant(1, pred_fake.shape)))
    print fake2.d.shape
    print x1.d.shape
    loss_vec = F.mean(F.squared_error(
        fake2, x1))
    fake_dis = fake.unlinked()
    pred_fake_dis = discriminator(fake_dis)
    loss_dis = F.mean(F.sigmoid_cross_entropy(
        pred_fake_dis, F.constant(0, pred_fake_dis.shape)))
    
    xBuf1 = nn.Variable([args.batch_size, 1, 56, 56])
    zBuf1 = vectorizer(xBuf1)
    xBuf2 = nn.Variable([args.batch_size, 1, 56, 56])
    zBuf2 = vectorizer(xBuf2)

    # Real path
    x = nn.Variable([args.batch_size, 1, 56, 56])
    pred_real = discriminator(x)
    loss_dis += F.mean(F.sigmoid_cross_entropy(pred_real,
                                               F.constant(1, pred_real.shape)))

    # Create Solver.
    solver_gen = S.Adam(args.learning_rate, beta1=0.5)
    solver_dis = S.Adam(args.learning_rate, beta1=0.5)
    solver_vec = S.Adam(args.learning_rate, beta1=0.5)
    with nn.parameter_scope("vec"):
        solver_vec.set_parameters(nn.get_parameters())
    with nn.parameter_scope("gen"):
        solver_vec.set_parameters(nn.get_parameters())
    with nn.parameter_scope("gen"):
        solver_gen.set_parameters(nn.get_parameters())
    with nn.parameter_scope("dis"):
        solver_dis.set_parameters(nn.get_parameters())

    # Create monitor.
    import nnabla.monitor as M
    monitor = M.Monitor(args.monitor_path)
    monitor_loss_gen = M.MonitorSeries("Generator loss", monitor, interval=10)
    monitor_loss_dis = M.MonitorSeries(
        "Discriminator loss", monitor, interval=10)
    monitor_loss_vec = M.MonitorSeries("Vectorizer loss", monitor, interval=10)
    monitor_time = M.MonitorTimeElapsed("Time", monitor, interval=100)
    monitor_fake = M.MonitorImageTile(
        "Fake images", monitor, normalize_method=lambda x: x + 1 / 2.)
    monitor_vec1 = M.MonitorImageTile(
        "vec images1", monitor, normalize_method=lambda x: x + 1 / 2.)
    monitor_vec2 = M.MonitorImageTile(
        "vec images2", monitor, normalize_method=lambda x: x + 1 / 2.)

    #data = data_iterator_mnist(args.batch_size, True)
    data = iterator.simple_data_iterator(load_kanji_data(),args.batch_size,True)

    # Training loop.
    for i in range(args.max_iter):
        if i % args.model_save_interval == 0:
            with nn.parameter_scope("gen"):
                nn.save_parameters(os.path.join(
                    args.model_save_path, "generator_param_%06d.h5" % i))
            with nn.parameter_scope("dis"):
                nn.save_parameters(os.path.join(
                    args.model_save_path, "discriminator_param_%06d.h5" % i))
            with nn.parameter_scope("vec"):
                nn.save_parameters(os.path.join(
                    args.model_save_path, "vectorizer_param_%06d.h5" % i))

        # Training forward
        image, _ = data.next()

        x1.d = image / 255. * 2 - 1.0
        # Generator update.
        solver_vec.zero_grad()
        loss_vec.forward(clear_no_need_grad=True)
        loss_vec.backward(clear_buffer=True)
        solver_vec.weight_decay(args.weight_decay)
        solver_vec.update()
        fake2.forward()
        monitor_vec1.add(i, fake2)
        monitor_vec2.add(i, x1)
        monitor_loss_vec.add(i, loss_vec.d.copy())

        image, _ = data.next()
        x.d = image / 255. * 2 - 1.0  # [0, 255] to [-1, 1]
        
        #z.d = np.random.randn(*z.shape)
        ratio = np.random.rand()
        image, _ = data.next()
        xBuf1.d = image / 255. * 2 - 1.0  # [0, 255] to [-1, 1]
        zBuf1.forward()
        
        image, _ = data.next()
        xBuf2.d = image / 255. * 2 - 1.0  # [0, 255] to [-1, 1]
        zBuf2.forward()
        z.d = (1-ratio) * zBuf1.d + ratio * zBuf2.d

        # Generator update.
        solver_gen.zero_grad()
        loss_gen.forward(clear_no_need_grad=True)
        loss_gen.backward(clear_buffer=True)
        solver_gen.weight_decay(args.weight_decay)
        solver_gen.update()
        monitor_fake.add(i, fake)
        monitor_loss_gen.add(i, loss_gen.d.copy())

        # Discriminator update.
        solver_dis.zero_grad()
        loss_dis.forward(clear_no_need_grad=True)
        loss_dis.backward(clear_buffer=True)
        solver_dis.weight_decay(args.weight_decay)
        solver_dis.update()
        monitor_loss_dis.add(i, loss_dis.d.copy())
        monitor_time.add(i)

    with nn.parameter_scope("gen"):
        nn.save_parameters(os.path.join(
            args.model_save_path, "generator_param_%06d.h5" % i))
    with nn.parameter_scope("dis"):
        nn.save_parameters(os.path.join(
            args.model_save_path, "discriminator_param_%06d.h5" % i))
    with nn.parameter_scope("vec"):
        nn.save_parameters(os.path.join(
            args.model_save_path, "vectorizer_param_%06d.h5" % i))


if __name__ == '__main__':
    monitor_path = 'tmp.monitor.gen'
    args = get_args(monitor_path=monitor_path, model_save_path=monitor_path,
                    max_iter=20000, learning_rate=0.0002, batch_size=64,
                    weight_decay=0.0001)
    train(args)
