#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

"""
In-memory dataset.
"""

import os
from os.path import join, exists

import numpy as np

from pgl.utils.data import Dataloader

from pahelix.utils.data_utils import save_data_list_to_npz, load_npz_to_data_list
from pahelix.utils.basic_utils import mp_pool_map


__all__ = ['InMemoryDataset']


class InMemoryDataset(object):
    """
    Description:
        The InMemoryDataset manages ``data_list`` which is a list of `data` and 
        the `data` is a dict of numpy ndarray. And each dict has the same keys.

        It works like a list: you can call `dataset[i] to get the i-th element of 
        the ``data_list`` and call `len(dataset)` to get the length of ``data_list``.
        
        The ``data_list`` can be cached in npz files by calling `dataset.save_data(data_path)` 
        and after that, call `InMemoryDataset(data_path)` to reload.


InMemoryDataset 管理着一个 data_list，它是一个包含 data 的列表，而 data 是一个字典，字典中的值都是 numpy 的 ndarray。并且每个字典具有相同的键。

它的工作方式类似于列表：你可以通过调用 dataset[i] 来获取 data_list 中的第 i 个元素，并通过调用 len(dataset) 来获取 data_list 的长度。

data_list 可以通过调用 dataset.save_data(data_path) 缓存到 npz 文件中，之后可以通过调用 InMemoryDataset(data_path) 来重新加载。

    Attributes:
        data_list(list): a list of dict of numpy ndarray.  #传入的是字典（smiles:label)

    Example:
        .. code-block:: python

            data_list = [{'a': np.zeros([4, 5])}, {'a': np.zeros([7, 5])}]
            dataset = InMemoryDataset(data_list=data_list)
            print(len(dataset))
            dataset.save_data('./cached_npz')   # save data_list to ./cached_npz

            dataset2 = InMemoryDataset(npz_data_path='./cached_npz')    # will load the saved `data_list`
            print(len(dataset))
    """
    def __init__(self, 
            data_list=None,
            npz_data_path=None,
            npz_data_files=None):
        """
        Users can either directly pass the ``data_list`` or pass the `data_path` from 
        which the cached ``data_list`` will be loaded.
#用户可以直接传递 data_list，或者传递 data_path，从中加载缓存的 data_list。
        Args:
            data_list(list): a list of dict of numpy ndarray.
            data_path(str): the path to the cached npz path.
        """
        super(InMemoryDataset, self).__init__()
        self.data_list = data_list
        self.npz_data_path = npz_data_path
        self.npz_data_files = npz_data_files

        if not npz_data_path is None:
            self.data_list = self._load_npz_data_path(npz_data_path)

        if not npz_data_files is None:
            self.data_list = self._load_npz_data_files(npz_data_files)

    def _load_npz_data_path(self, data_path):
        data_list = []
        files = [f for f in os.listdir(data_path) if f.endswith('.npz')]
        files = sorted(files)
        for f in files:
            data_list += load_npz_to_data_list(join(data_path, f))
        return data_list

    def _load_npz_data_files(self, data_files):
        data_list = []
        for f in data_files:
            data_list += load_npz_to_data_list(f)
        return data_list

    def _save_npz_data(self, data_list, data_path, max_num_per_file=10000):
        if not exists(data_path):
            os.makedirs(data_path)
        n = len(data_list)
        for i in range(int((n - 1) / max_num_per_file) + 1):
            filename = 'part-%06d.npz' % i
    # %06d 是格式化字符串，用于将 i 格式化为一个 6 位数的字符串，不足 6 位的部分用前导零填充。例如，i 为 3 时，文件名将为 part-000003.npz。
            sub_data_list = self.data_list[i * max_num_per_file: (i + 1) * max_num_per_file]
            save_data_list_to_npz(sub_data_list, join(data_path, filename))

    def save_data(self, data_path):
        """
        Save the ``data_list`` to the disk specified by ``data_path`` with npz format.
        After that, call `InMemoryDataset(data_path)` to reload the ``data_list``.

        Args:
            data_path(str): the path to the cached npz path.
        """
        self._save_npz_data(self.data_list, data_path)

    def __getitem__(self, key):  #不是很明白，好像是key是切片，整数，列表，放回对应的数据
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            dataset = InMemoryDataset(
                    data_list=[self[i] for i in range(start, stop, step)])
            return dataset
        elif isinstance(key, int) or \
                isinstance(key, np.int64) or \
                isinstance(key, np.int32):
            return self.data_list[key]
        elif isinstance(key, list):
            dataset = InMemoryDataset(
                    data_list=[self[i] for i in key])
            return dataset
        else:
            raise TypeError('Invalid argument type: %s of %s' % (type(key), key))

    def __len__(self):
        return len(self.data_list)

    def transform(self, transform_fn, num_workers=4, drop_none=False):
        """
        Inplace apply `transform_fn` on the `data_list` with multiprocess.
        """
        data_list = mp_pool_map(self.data_list, transform_fn, num_workers)  #mp_pool_map 函数的作用是利用多进程并行处理 self.data_list 中的每个数据项，以提高处理效率
        if drop_none:
            self.data_list = [data for data in data_list if not data is None]  #去掉None
        else:
            self.data_list = data_list


    def get_data_loader(self, batch_size, num_workers=4, shuffle=False, collate_fn=None, drop_last=False):
        """
        It returns an batch iterator which yields a batch of data. Firstly, a sub-list of
        `data` of size ``batch_size`` will be draw from the ``data_list``, then 
        the function ``collate_fn`` will be applied to the sub-list to create a batch and 
        yield back. This process is accelerated by multiprocess.

        Args:
            batch_size(int): the batch_size of the batch data of each yield.
            num_workers(int): the number of workers used to generate batch data. Required by 
                multiprocess.
            shuffle(bool): whether to shuffle the order of the ``data_list``.
            collate_fn(function): used to convert the sub-list of ``data_list`` to the 
                aggregated batch data. #用于将data_list的子列表转换为聚合的批次数据。

        Yields:
            the batch data processed by ``collate_fn``.
        """
        return Dataloader(self, 
                batch_size=batch_size, 
                num_workers=num_workers, 
                shuffle=shuffle,
                drop_last=drop_last,
                collate_fn=collate_fn)


#drop_last 是一个布尔值，用于指定在数据集大小不能被批次大小整除时，是否丢弃最后一个不完整的批次。
如果设置为 True，那么当最后一个批次不足 batch_size 时，将不会返回这个批次；如果 False，则会返回这个批次（可能大小小于 batch_size）。

collate_fn 是一个可选的函数，用于定义如何将来自数据集的样本合并为一个批次。
默认情况下，PyTorch 和 pgl 提供的 collate_fn 可以处理常见的数据类型，但如果数据结构更复杂，可以自定义这个函数来处理特定的情况。
