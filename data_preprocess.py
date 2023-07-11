import os
import random
import shutil


def partition(types, root):
    """
    按0.7，0.2，0.1的比例划分数据集
    :param types:
    :param root:
    :return:
    """
    data = {}
    # 为各类型添加数据
    for type in types:
        data[type] = []
        for file in os.listdir(root):
            if file.split("_")[0] == type and not (file.split(".")[-2] in data[type]):
                data[type].append(file.split(".")[-2])

    train = []
    valid = []
    test = []
    for k in data.keys():
        num = len(data[k])  # 该部分的文件总数
        # 打乱下标
        random.shuffle(data[k])
        train_num = int(num * 0.7)
        valid_num = int(num * 0.2)
        train += data[k][:train_num]
        valid += data[k][train_num:train_num + valid_num]
        test += data[k][train_num + valid_num:]
    return train, valid, test


def creat_data(root, train, valid, test):
    """
    将数据提取到dataset文件夹下
    :param root:
    :param train:
    :param valid:
    :param test:
    :return:
    """
    destination_folder = 'dataset'

    # 自动创建输出目录
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        os.mkdir(destination_folder + "/train")
        os.mkdir(destination_folder + "/valid")
        os.mkdir(destination_folder + "/test")

    for file in os.listdir(root):
        name = file.split(".")[-2]
        if name in train:
            destination_path = destination_folder + "/train"
        elif name in valid:
            destination_path = destination_folder + "/valid"
        elif name in test:
            destination_path = destination_folder + "/test"
        else:
            continue
        png = name + ".png"
        json = name + ".json"
        # 排除图片和json不缺少的情况
        if png in os.listdir(root) and json in os.listdir(root):
            shutil.copy(os.path.join(root, png), os.path.join(destination_path, png))
            shutil.copy(os.path.join(root, json), os.path.join(destination_path, json))


if __name__ == "__main__":
    root = "hist_cell_images"
    types = []
    # 获取所有部位的类别
    for file in os.listdir(root):
        type = file.split("_")[0]
        types.append(type)
    types = set(types)

    # 获得属于部位的各文件字典
    train, valid, test = partition(types, root)
    creat_data(root, train, valid, test)
