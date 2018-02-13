# coding=utf-8
import os
import os.path as osp
from django.conf import settings
import re

import numpy as np


labels = None
files = None


def get_all_labels():
    """
    获取所有类别
    :return:
    """
    global labels
    # 如果是第一次进入函数
    if labels is None:
        # 列出文件夹下所有子文件夹名称作为类别
        labels = os.listdir(osp.join(settings.BASE_DIR, "ImagesDB"))
        labels = [{"label": label, "num": len(os.listdir(osp.join(osp.join(settings.BASE_DIR, "ImagesDB"), label)))} for label in labels]
    return labels


def get_all_images():
    """
    列出数据库中所有图片文件
    :return:
    """
    global files
    if files is None:
        labels = [item["label"] for item in get_all_labels()]
        files = []
        prefix = osp.join(settings.BASE_DIR, "ImagesDB")
        for label in labels:
            x = os.listdir(osp.join(prefix, label))
            x = list(map(lambda item: "".join([r"/static/", label, "/", item]), x))
            files = np.hstack((files, x))
    return files


def write2disk(file_path, file_data):
    """
    将文件内容file_data写入路径file_path中
    :param file_path: 目标路径
    :param file_data: 文件内容
    :return:
    """
    dir_name = osp.dirname(file_path)
    if not osp.exists(dir_name):
        os.mkdir(dir_name)
    with open(file_path, "wb+") as f:
        for chunk in file_data.chunks():
            f.write(chunk)


def resolve_file_name(file_path):
    """
    从文件路径中解析出文件名
    :param file_path:
    :return: 图片类别，图片文件名
    """
    label, filename = file_path.split("/")
    return label, filename


def resolve_report(report_string):
    """
    从准确率统计字符串中解析出各个类别对应的准确率
    :param report_string: 格式如下：
                precision    recall  f1-score   support

          0       1.00      0.99      0.99        88
          1       0.99      0.97      0.98        91
          2       0.99      0.99      0.99        86
          3       0.98      0.87      0.92        91
          4       0.99      0.96      0.97        92
          5       0.95      0.97      0.96        91
          6       0.99      0.99      0.99        91
          7       0.96      0.99      0.97        89
          8       0.94      1.00      0.97        88
          9       0.93      0.98      0.95        92

    avg / total       0.97      0.97      0.97       899
    :return: 该字符串对应的二维矩阵
    """
    # print(report_string)
    rows = report_string.split("\n")
    r = [re.split(r"\s+", rows[0], re.S)[-5:-1]]
    for row in rows[2:-3]:
        r.append(re.split(r"\s+", row, re.S)[-5:-1])
    r.append(re.split(r"\s+", rows[-2], re.S)[-5:-1])
    r[-1][0] = "avg/total"
    return r
