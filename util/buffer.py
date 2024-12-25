# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from typing import Tuple


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size


class Buffer:
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, device, n_tasks=1, mode='reservoir', attr_num=3):
        assert mode in ['ring', 'reservoir']
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.functional_index = eval(mode)
        if mode == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        self.attr_num = attr_num
        self.buffer = []

    def init_tensors(self, *batch) -> None:
        """
        Initializes just the required tensors.
        """
        for attr in batch:
            self.buffer.append(torch.zeros((self.buffer_size, *attr.shape[1:]), dtype=torch.float32, device=self.device))

    def add_data(self, *batch):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """
        if self.num_seen_examples == 0:
            self.init_tensors(*batch)

        for i in range(batch[0].shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                for j, attr in enumerate(batch):
                    self.buffer[j][index] = attr.detach().to(self.device)

    def get_data(self, size: int) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.num_seen_examples, self.buffer[0].shape[0]):
            size = min(self.num_seen_examples, self.buffer[0].shape[0])

        choice = np.random.choice(min(self.num_seen_examples, self.buffer[0].shape[0]),
                                  size=size, replace=False)
        rets = []
        for attr in self.buffer:
            rets += [attr[choice]]
        return rets

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        return tuple(self.buffer)

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        self.buffer = []
        self.num_seen_examples = 0

