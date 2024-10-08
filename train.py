import numpy as np
import paddle
from prettytable import prettytable
from sklearn.metrics import matthews_corrcoef, f1_score, recall_score, precision_score, accuracy_score
from src.utils import calc_rocauc_score
from utils import specificityCalc


def training(args, model, train_dataset, collate_fn, criterion, encoder_opt, head_opt):
    """
    Define the train function
    Args:
        args,model,train_dataset,collate_fn,criterion,encoder_opt,head_opt;
    Returns:
        the average of the list loss
    """
    data_gen = train_dataset.get_data_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_fn)
    total_pred = []
    total_label = []
    total_valid = []
    total_graph = []
    list_loss = []
    model.train()
    for atom_bond_graphs, bond_angle_graphs, valids, labels in data_gen:
        if len(labels) < args.batch_size * 0.5:  #如果批次不足设定批次大小的一般，舍弃
            continue
        atom_bond_graphs = atom_bond_graphs.tensor()
        bond_angle_graphs = bond_angle_graphs.tensor()
        labels = paddle.to_tensor(labels, 'float32')
        valids = paddle.to_tensor(valids, 'float32')
        # if flag:
        perturb1 = paddle.to_tensor(
            paddle.uniform([atom_bond_graphs.node_feat['atomic_num'].shape[0], 32], min=-1e-5, max=1e-5),
            stop_gradient=False)
        perturb2 = paddle.to_tensor(
            paddle.uniform([atom_bond_graphs.edge_feat['bond_dir'].shape[0], 32], min=-1e-5, max=1e-5),
            stop_gradient=False)
        
#生成了一个形状为 [节点数, 32] 的二维张量，张量中的每个元素都是从 [-1e-5, 1e-5] 区间内的均匀分布随机数。
#在 PaddlePaddle 中，stop_gradient 参数用于控制是否在反向传播时计算梯度。
#设置 stop_gradient=False 表示该张量在反向传播时会计算梯度，这意味着这个张量会参与到梯度计算中，对应的梯度会被更新
        
        preds, graph_repr = model(atom_bond_graphs, bond_angle_graphs, perturb1,perturb2)
        loss = criterion(preds, labels)
        loss /= 4  #为什么？
        
#这个迭代主要是用梯度更新扰动值，不直接*梯度，防止梯度值过大，造成扰动值过大
        for _ in range(3):
            loss.backward()
            perturb_data1 = perturb1 + 1e-3 * paddle.sign(perturb1.grad)
            perturb1.set_value(perturb_data1)
            perturb1.clear_grad()
            perturb_data2 = perturb2 + 1e-3 * paddle.sign(perturb2.grad)
            perturb2.set_value(perturb_data2)
            perturb2.clear_grad()
            preds, graph_repr = model(atom_bond_graphs, bond_angle_graphs, perturb1, perturb2)
            loss = criterion(preds, labels)
            loss /= 4
#paddle.sign 的工作原理很简单：检查每个元素的值：如果元素大于0，结果是1。如果元素小于0，结果是-1。如果元素等于0，结果是0。

        loss.backward()
        head_opt.step()
        head_opt.clear_grad()
        encoder_opt.step()
        encoder_opt.clear_grad()

        list_loss.append(loss.numpy()*4)
        total_pred.append(preds.numpy())
        total_label.append(labels.numpy())
        total_valid.append(valids.numpy())
        total_graph.append(graph_repr.numpy())
    total_pred = np.concatenate(total_pred, 0).round().astype('int32')  #用来四舍五入函数来处理预测的概率大小
    total_label = np.concatenate(total_label, 0).astype('int32')
    total_valid = np.concatenate(total_valid, 0)
    total_graph = np.concatenate(total_graph, 0)
    auc = calc_rocauc_score(total_label, total_pred, total_valid)
    acc = accuracy_score(total_label, total_pred)
    precison = precision_score(total_label, total_pred)
    recall = recall_score(total_label, total_pred)
    f1 = f1_score(total_label, total_pred)
    specificity = specificityCalc(total_label, total_pred)[1]
    mcc = matthews_corrcoef(total_label, total_pred)

    table = prettytable.PrettyTable()
    table.field_names = ['Loss', 'AUC', 'Accuracy', 'Precison', 'Recall', 'F1_score', 'Specificity', 'MCC']
    table.add_row(
        [np.mean(list_loss), auc, acc, precison, recall, f1,
         specificity, mcc])
    return np.mean(list_loss), auc, table, total_graph, total_label


def evaluate(args, model, test_dataset, collate_fn):
    """
    Define the evaluate function
   `valid` ten  In the dataset, a proportion of labels are blank. So we use asor
    to help eliminate these blank labels in both training and evaluation phase.
    """
    data_gen = test_dataset.get_data_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn)
    total_pred = []
    total_label = []
    list_loss = []
    total_valid = []
    total_graph = []

    model.eval()
    for atom_bond_graphs, bond_angle_graphs, valids, labels in data_gen:
        atom_bond_graphs = atom_bond_graphs.tensor()
        bond_angle_graphs = bond_angle_graphs.tensor()
        labels = paddle.to_tensor(labels, 'float32')
        valids = paddle.to_tensor(valids, 'float32')
        preds, graph_repr = model(atom_bond_graphs, bond_angle_graphs)
        list_loss.append(paddle.nn.BCELoss()(preds, labels).numpy())
        total_pred.append(preds.numpy())
        total_valid.append(valids.numpy())
        total_label.append(labels.numpy())
        total_graph.append(graph_repr.numpy())

    total_pred = np.concatenate(total_pred, 0)
    total_label = np.concatenate(total_label, 0)
    total_valid = np.concatenate(total_valid, 0)
    total_graph = np.concatenate(total_graph, 0)
    auc = calc_rocauc_score(total_label, total_pred, total_valid)
    acc = accuracy_score(total_label, total_pred.round())
    precison = precision_score(total_label, total_pred.round())
    recall = recall_score(total_label, total_pred.round())
    f1 = f1_score(total_label, total_pred.round())
    specificity = specificityCalc(total_label, total_pred.round())[1]
    mcc = matthews_corrcoef(total_label, total_pred.round())

    table = prettytable.PrettyTable()
    table.field_names = ['Loss', 'AUC', 'Accuracy', 'Precison', 'Recall', 'F1_score', 'Specificity', 'MCC']
    table.add_row(
        [np.mean(list_loss), auc, acc, precison, recall, f1,
         specificity, mcc])
    return np.mean(list_loss), auc, table, total_graph, total_label, total_pred,mcc
