#!/usr/bin/python
#-*-coding:utf-8-*-
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
Processing of Blood-Brain Barrier Penetration dataset

The Blood-brain barrier penetration (BBBP) dataset is extracted from a study on the modeling and 
prediction of the barrier permeability. As a membrane separating circulating blood and brain extracellular fluid, the blood-brain barrier blocks most drugs, hormones and neurotransmitters. Thus penetration of the barrier forms a long-standing issue in development of drugs targeting central nervous system.
This dataset includes binary labels for over 2000 compounds on their permeability properties.

血脑屏障穿透数据集处理

血脑屏障穿透（BBBP）数据集来源于对屏障通透性的建模和预测研究。血脑屏障作为分隔循环血液与脑细胞外液的膜，阻挡了大多数药物、激素和神经递质。因此，突破这一屏障一直是针对中枢神经系统的药物开发中的一个长期问题。

该数据集包含了2000多种化合物的二分类标签，用于描述它们的穿透性特征。
You can download the dataset from
http://moleculenet.ai/datasets-1 and load it into pahelix reader creators
"""

import os
from os.path import join, exists       
import pandas as pd
import numpy as np

from pahelix.datasets.inmemory_dataset import InMemoryDataset


__all__ = ['get_default_bbbp_task_names', 'load_bbbp_dataset']      #是 Python 模块中的一个特殊变量，用于定义当前模块中可以被 from module import * 语句导入的公共接口。

（一个下划线的话就表示只在内部使用）

def get_default_bbbp_task_names(): 
    """Get that default bbbp task names and return the binary labels"""
    return ['p_np']


def load_bbbp_dataset(data_path, task_names=None):   #对BBBP数据集文件进行初始文件加载处理
    """Load bbbp dataset ,process the classification labels and the input information.

    Description:

        The data file contains a csv table, in which columns below are used:
            
            Num:number
            
            name:Name of the compound
            
            smiles:SMILES representation of the molecular structure
            
            p_np:Binary labels for penetration/non-penetration（穿透/非穿透）

    Args:
        data_path(str): the path to the cached npz path.  #缓存的 npz 文件路径。
        task_names(list): a list of header names to specify the columns to fetch from   #一个列名称的列表，用于指定从 CSV 文件中提取的列 (这是对应的是否有肝毒性的标签吗）
        目前看来只是（穿透/非穿透） 的标签
            the csv file.
    
    Returns:
        an InMemoryDataset instance.
    
    Example:
        .. code-block:: python

            dataset = load_bbbp_dataset('./bbbp')
            print(len(dataset))
        
    References:
    
    [1] Martins, Ines Filipa, et al. “A Bayesian approach to in silico blood-brain barrier penetration modeling.” Journal of chemical information and modeling 52.6 (2012): 1686-1697.
    
    """

    if task_names is None:  #task_names 就是该分子的标签分类（目前看来如果，不指定标签文件，便会使用该数据库本身的 p_np标签
        task_names = get_default_bbbp_task_names()

    raw_path = join(data_path, 'raw')
    csv_file = os.listdir(raw_path)[0]   #listdir 是os模块中的一个函数，用于列出指定路径（raw_path）下的所有文件和目录名（选取列表第一个文件的代码过于粗糙，具体应用需要修改）
    input_df = pd.read_csv(join(raw_path, csv_file), sep=',')
    smiles_list = input_df['smiles']
    
    from rdkit.Chem import AllChem
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    
    preprocessed_rdkit_mol_objs_list = [m if not m is None else None for m in   rdkit_mol_objs_list]  #如果分子不能顺利转为rdkit对象，会在列表中添加None
    
    smiles_list = [AllChem.MolToSmiles(m) if not m is None else None for m in preprocessed_rdkit_mol_objs_list]

    #RDKit 在将 SMILES 转换为分子对象时，会尝试对其进行标准化。这意味着，即使两个 SMILES 字符串描述的是相同的分子，它们的表示形式也可能略有不同（例如，原子顺序、环闭合方式等）。
通过转换为 RDKit 分子对象并再次转回 SMILES，可以确保所有的 SMILES 字符串都遵循相同的标准化规则，这对于后续的数据处理和分析可能是有益的。
    
    labels = input_df[task_names]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans  #为啥要转为-1尼？）

    data_list = []
    for i in range(len(smiles_list)):
        if smiles_list[i] is None:
            continue
        data = {}
        data['smiles'] = smiles_list[i]        
        data['label'] = labels.values[i]
        data_list.append(data)
    dataset = InMemoryDataset(data_list)
    return dataset  #只放回了分子的smile格式的列表，那标签 咋办尼？
