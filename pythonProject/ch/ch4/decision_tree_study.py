import math
import pandas as pd
import numpy as np
#import operator
import tree_plotter


def load_dataset(filename):
    df = pd.read_csv(filename)
    dataset = np.array(df.iloc[:]).tolist()
    feat_names = list(df.columns)
    return dataset, feat_names


def calc_ent(dataset):
    '''计算信息熵'''
    ent = 0.0
    return ent


def extract_dataset(dataset, axis, value):
    '''
    抽取数据集dataset的第axis属性等于value的样本, 构建子数据集
    '''
    sub_dataset = []
    for example in dataset:
        if example[axis] == value:
            sub_example = example[: axis]
            sub_example.extend(example[axis+1:])
            sub_dataset.append(sub_example)
    return sub_dataset


def choose_best_feature(dataset):
    num_features = len(dataset[0]) - 1
    base_ent = calc_ent(dataset)
    best_info_gain = 0.0
    best_feature = -1
    # 遍历所有的特征
    for i in range(num_features):
        # 创建所有样本的第i个特征列表
        feat_list = [example[i] for example in dataset]
        # 将特征表转换成集合, 合并列表中重复的特征取值
        feat_set = set(feat_list)
        new_ent = 0.0
        for value in feat_set:
            sub_dataset = extract_dataset(dataset, i, value)
            prob = len(sub_dataset) / float(len(dataset))
            new_ent += prob * calc_ent(sub_dataset)
        # 计算信息增益
        info_gain = base_ent - new_ent
        # 记录最大信息增益和对应的特征
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i

    return best_feature


def majority_cnt(class_list):
    """寻找class_list中最多的类别, 并返回类别号"""
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1

    # 将字典表项按照计数大小反排序
    #sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    sorted_class_count = sorted(class_count.items(), key=lambda item: item[1], reverse=True)
    # 返回样本数最多的类别
    return sorted_class_count[0][0]


def create_decision_tree(dataset, feat_names):
    """
    创建决策树, 决策树用字典表示, {属性名: {属性值1: 类别|子节点, 属性值2: 类别|子节点, ...}}
    :param dataset:         数据集, 二维表, 存储m个样本的n个属性值和样本的类别
    :param feat_names:      特征属性名称列表
    :return:                决策树
    """
    pass
    

def classify(tree, feat_names, input):
    """
    决策树分类
    :param tree:            决策树
    :param feat_names:      特征名列表
    :param input:           输入特征
    :return:                分类值
    """
    keys = tuple(tree.keys())
    first_attr = keys[0]
    sub_dict = tree[first_attr]
    feat_index = feat_names.index(first_attr)
    key = input[feat_index]
    value = sub_dict[key]
    if isinstance(value, dict):
        label = classify(value, feat_names, input)
    else:
        label = value
    return label


if __name__ == '__main__':

    dataset, feat_names = load_dataset('watermelon.csv')
    tree = create_decision_tree(dataset, feat_names.copy())
    #print(tree)

    for example in dataset:
        pred = classify(tree, feat_names, example)
        print('原始值： ', example[-1], ', 预测值： ', pred)

    tree_plotter.createPlot(tree)
