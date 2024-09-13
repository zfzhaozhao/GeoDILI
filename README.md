# GeoDILI: a robust and interpretable graph neural network for Drug-Induced Liver Injury prediction using molecular geometric representation
GeoDILI：一种用于药物诱导肝损伤预测的鲁棒且可解释的图神经网络模型，基于分子几何表示
# Background
In this study, we developed a highly accurate and interpretable human DILI prediction model named GeoDILI. An overview of the proposed model is shown in following figure:  
在这项研究中，我们开发了一种高精度且可解释的人类药物诱导肝损伤（DILI）预测模型，命名为 GeoDILI。下面的图示展示了该模型的概述
![image](image.jpg)

The GeoDILI model used a pre-trained 3D spatial structure-based GNN to extract molecular representations, followed by a residual neural network to make an accurate DILI prediction. The gradient information from the final graph convolutional layer of GNN was utilized to obtain atom-based weights, which enabled the identification of dominant substructures that significantly contributed to DILI prediction. We evaluated the performance of GeoDILI by comparing it with the SOTA DILI prediction tools, popular GNN models, as well as conventional Deep Neural Networks (DNN) and ML models, confirming its effectiveness in predicting DILI. In addition, we applied our model to three different human DILI datasets from various sources, namely DILIrank, DILIst, and a dataset recently collected by Yan et al.. Results showed performance differences across datasets and suggested that a smaller, high-quality dataset DILIrank may lead to better results. Finally, we applied the dominant substructure inference method to analyze the entire DILIrank dataset and identified seven significant SAs with both high precision and potential mechanisms. 
GeoDILI 模型使用了预训练的基于 3D 空间结构的图神经网络（GNN）来提取分子表示，随后通过残差神经网络进行精确的 DILI 预测。我们利用 GNN 最终图卷积层的梯度信息来获得原子级别的权重，从而识别出对 DILI 预测具有重要贡献的主要子结构。我们通过与最先进的 DILI 预测工具、流行的 GNN 模型以及传统的深度神经网络（DNN）和机器学习（ML）模型进行比较，评估了 GeoDILI 的性能，确认了其在 DILI 预测中的有效性。此外，我们将模型应用于来自不同来源的三个人类 DILI 数据集，即 DILIrank、DILIst 和 Yan 等人最近收集的数据集。结果显示不同数据集之间存在性能差异，且较小但高质量的数据集 DILIrank 可能会带来更好的结果。最后，我们应用主要子结构推断方法分析了整个 DILIrank 数据集，并识别出七个具有高精度和潜在机制的重要子结构（SA）。
# Installation guide
## Prerequisites

* OS support: Windows, Linux
* Python version: 3.6, 3.7, 3.8

## Dependencies

| name         | version |
|   ------------   |   ----   |
|      pandas      | \==1.3.5 |
|     networkx     | \==2.6.3 |
| paddlepaddle-gpu | \==2.0.0 |
|       pgl        | \==2.2.4 |
|    rdkit-pypi    | \==2022.3.5 |
|     sklearn      | \==1.0.2 |
|      tqdm        | \==4.64.0 |
|   prettytable    | \==3.4.1 |
|    matplotlib    | \==3.5.2 |

Please use the following environment installation command:

    $ pip3 install -r requirements.txt

Note that the requirements.txt file does not contain the command to install paddlepaddle-gpu, you need to run the following command to install it separately:

    $ python -m pip install paddlepaddle-gpu==2.3.2.post112 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html

All datas for the experiment are contained in the [DILI.zip](DILI.zip) file.

# Usage

To train a model with an existing dataset:

    $ python main.py --dataset_name dilirank --task train --split_type random
    $ python main.py --dataset_name rega --task train --split_type random
    $ python main.py --dataset_name diliset --task train --split_type random
    $ python main.py --dataset_name bbbp --task train --split_type random

To test with an existing model:

    $ python main.py --dataset_name dilirank --task test --split_type random
    $ python main.py --dataset_name rega --task test --split_type random

## Result

|     Dataset      |    AUC    |    ACC    |    MCC    |
| :--------------: | :-------: | :-------: | :-------: |
|     DILIrank     | **0.908** | **0.875** | **0.732** |
|      DILIst      | **0.851** | **0.786** | **0.553** |
| Yan et al (rega) | **0.843** | **0.773** | **0.549** |

|                           DILIrank                           |                     Yan et al(rega)                     |
| :----------------------------------------------------------: | :-----------------------------------------------------: |
| ![dilirank](dilirank.png) | ![raga](raga.png) |



## Citation

If you use the code or data in this package, please cite:

```bibtex
Wenxuan Wu, Jiayu Qian, Changjie Liang, Jingya Yang, Guangbo Ge, Qingping Zhou, and Xiaoqing Guan
Chemical Research in Toxicology 2023 36 (11), 1717-1730
DOI: 10.1021/acs.chemrestox.3c00199
```
