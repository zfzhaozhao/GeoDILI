#!/usr/bin/python                                                                                                
# -*-coding:utf-8-*-
#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
gnn network  #构建图神经网络模块，是下游任务的网络
"""

import paddle
import paddle.nn as nn
import pgl

from pahelix.networks.basic_block import MLP, Residual  #使用了残差网络


class DownstreamModel(nn.Layer):
    """
    Docstring for DownstreamModel,it is an supervised 
    GNN model which predicts the tasks shown in num_tasks and so on.
    #这是一个监督学习的图神经网络（GNN）模型，用于预测在 num_tasks 中显示的任务等。
    """

    def __init__(self, model_config, compound_encoder):
        super(DownstreamModel, self).__init__()
        self.task_type = model_config['task_type']
        self.num_tasks = model_config['num_tasks']
#model_config 是一个用来记录了在模型中要用到的配置，和超参数
        
        self.compound_encoder = compound_encoder
        #这个嵌入层是是传入的函数，得关注一下
        self.norm = nn.LayerNorm(compound_encoder.graph_dim)
        self.hid = 256
        # self.mlp = MLP(
        #         model_config['layer_num'],
        #         in_size=compound_encoder.graph_dim,
        #         hidden_size=model_config['hidden_size'],
        #         out_size=self.num_tasks,
        #         act=model_config['act'],
        #         dropout_rate=model_config['dropout_rate'])
        # self.mlp = MLP(
        #         3,
        #         in_size=compound_encoder.graph_dim,
        #         hidden_size=128,
        #         out_size=self.num_tasks,
        #         act=model_config['act'],
        #         dropout_rate=model_config['dropout_rate'])
        self.resblock_dense = nn.Sequential(nn.Linear(compound_encoder.graph_dim, self.hid),
                                            Residual(self.hid, self.hid),
                                            Residual(self.hid, self.hid),
                                            Residual(self.hid, self.hid),
                                            Residual(self.hid, self.hid),
                                            nn.Linear(self.hid, self.num_tasks),
                                            nn.Sigmoid()
                                            )

    def forward(self, atom_bond_graphs, bond_angle_graphs, perturb1=0,perturb2=0):
        """
        Define the forward function,set the parameter layer options.compound_encoder 
        creates a graph data holders that attributes and features in the graph.
        Returns:
            pred: the model prediction.
        """
        node_repr, edge_repr, graph_repr = self.compound_encoder(atom_bond_graphs, bond_angle_graphs, perturb1, perturb2)
        graph_repr = self.norm(graph_repr)
        pred = self.resblock_dense(graph_repr)
        # pred = self.mlp(graph_repr)

        return pred, graph_repr
