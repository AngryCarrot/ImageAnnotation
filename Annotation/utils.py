import os
import os.path as osp
from django.conf import settings
import re

import numpy as np


labels = None
files = None


def get_all_labels():
    global labels
    if labels is None:
        labels = os.listdir(osp.join(settings.BASE_DIR, "ImagesDB"))
        labels = [{"label": label, "num": len(os.listdir(osp.join(osp.join(settings.BASE_DIR, "ImagesDB"), label)))} for label in labels]
    return labels


def get_all_images():
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
    dir_name = osp.dirname(file_path)
    if not osp.exists(dir_name):
        os.mkdir(dir_name)
    with open(file_path, "wb+") as f:
        for chunk in file_data.chunks():
            f.write(chunk)


def resolve_file_name(file_path):
    a = file_path.split("/")
    return a[0], a[1]


def resolve_report(report_string):
    print(report_string)
    rows = report_string.split("\n")
    r = [re.split(r"\s+", rows[0], re.S)[-5:-1]]
    for row in rows[2:-3]:
        r.append(re.split(r"\s+", row, re.S)[-5:-1])
    r.append(re.split(r"\s+", rows[-2], re.S)[-5:-1])
    r[-1][0] = "avg/total"
    return r
