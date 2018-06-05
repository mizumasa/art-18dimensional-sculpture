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

'''
Provide data iterator for (images,labels).
'''
import numpy
import struct

from nnabla.logger import logger
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource
from nnabla.utils.data_source_loader import download

class SimpleDataSource(DataSource):
    def _get_data(self, position):
        image = self._images[self._indexes[position]]
        label = self._labels[self._indexes[position]]
        return (image, label)

    def __init__(self, _data ,train=True, shuffle=False, rng=None):
        super(SimpleDataSource, self).__init__(shuffle=shuffle)
        self._train = train
        # With python3 we can write this logic as following, but with
        # python2, gzip.object does not support file-like object and
        # urllib.request does not support 'with statement'.
        #
        #   with request.urlopen(label_uri) as r, gzip.open(r) as f:
        #       _, size = struct.unpack('>II', f.read(8))
        #       self._labels = numpy.frombuffer(f.read(), numpy.uint8).reshape(-1, 1)
        #
        inputs,outputs = _data
        self._labels = outputs
        self._images = inputs
        
        self._size = len(self._labels)
        self._variables = ('x', 'y')
        if rng is None:
            rng = numpy.random.RandomState(313)
        self.rng = rng
        self.reset()

    def reset(self):
        if self._shuffle:
            self._indexes = self.rng.permutation(self._size)
        else:
            self._indexes = numpy.arange(self._size)
        super(SimpleDataSource, self).reset()

    @property
    def images1(self):
        """Get copy of whole data with a shape of (N, 1, H, W)."""
        return self._images1.copy()

    @property
    def labels(self):
        """Get copy of whole label with a shape of (N, 1)."""
        return self._labels.copy()

def simple_data_iterator(_data,batch_size,
                        train=True,
                        rng=None,
                        shuffle=True,
                        with_memory_cache=False,
                        with_parallel=False,
                        with_file_cache=False):
    return data_iterator(SimpleDataSource(_data, train=train, shuffle=shuffle, rng=rng),
                         batch_size,
                         with_memory_cache,
                         with_parallel,
                         with_file_cache)
