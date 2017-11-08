# coding=utf-8
from django.shortcuts import render
from django.http import JsonResponse
from django.http import HttpResponse
from django.conf import settings
from django.template.loader import render_to_string

import os
import os.path as osp
import shutil
import time
import numpy as np

from .forms import TrainForm
from .forms import SearchForm
from .utils import get_all_labels
from .utils import write2disk
from .utils import get_all_images
from .utils import resolve_file_name
from .utils import resolve_report


# Create your views here.
def index(request):
    return render(request, 'index.html', {"Labels": get_all_labels(), "SearchForm": SearchForm()})


def gallery(request, label):
    file_list = os.listdir(osp.join(osp.join(settings.BASE_DIR, "ImagesDB"), label))
    file_list = list(map(lambda item: r"/static/" + label + r"/" + item, file_list))
    return render(request, 'gallery.html', {"File": file_list, "Labels": get_all_labels(), "SearchForm": SearchForm()})


def gallery2(request):
    return render(request, 'index.html', {"File": get_all_labels(), "SearchForm": SearchForm()})


def slider(request):
    files = get_all_images()
    # print(files.shape)
    return render(request, 'slider.html', {"Labels": get_all_labels(), "File": files, "SearchForm": SearchForm()})


def overview(request):
    return render(request, 'overview.html', {"Labels": get_all_labels(), "SearchForm": SearchForm()})


def upload(request, is_single):
    if is_single == "single":
        return render(request, 'upload_s.html', {"Labels": get_all_labels(), "SearchForm": SearchForm()})
    else:
        return render(request, 'upload_m.html', {"Labels": get_all_labels(), "SearchForm": SearchForm()})


def classify(request):
    return render(request, "classify.html", {"Labels": get_all_labels(), "SearchForm": SearchForm()})


def classify_result(request):
    return render(request, "classify.html", {"Labels": get_all_labels(), "SearchForm": SearchForm()})
    # result = process_classify()
    # return render(request, "results_cls.html", {"Labels": get_all_labels(), "Results": result})


def train(request):
    form = TrainForm()
    return render(request, "train.html", {"Labels": get_all_labels(), "form": form, "SearchForm": SearchForm()})


def validate(request):
    return render(request, "validate.html", {"Labels": get_all_labels(), "SearchForm": SearchForm()})


def dashboard(request):
    return render(request, 'dashboard.html', {"Labels": get_all_labels(), "SearchForm": SearchForm()})


def grids(request):
    return render(request, 'grids.html', {"Labels": get_all_labels(), "SearchForm": SearchForm()})


def media(request):
    return render(request, 'media.html', {"Labels": get_all_labels(), "SearchForm": SearchForm()})


def general(request):
    return render(request, 'general.html', {"Labels": get_all_labels(), "SearchForm": SearchForm()})


def typography(request):
    return render(request, 'typography.html', {"Labels": get_all_labels(), "SearchForm": SearchForm()})


def widgets(request):
    return render(request, 'widgets.html', {"Labels": get_all_labels(), "SearchForm": SearchForm()})


def inbox(request):
    return render(request, 'inbox.html', {"Labels": get_all_labels(), "SearchForm": SearchForm()})


def compose(request):
    return render(request, 'compose.html', {"Labels": get_all_labels(), "SearchForm": SearchForm()})


def tables(request):
    return render(request, 'tables.html', {"Labels": get_all_labels(), "SearchForm": SearchForm()})


def forms(request):
    return render(request, 'forms.html', {"Labels": get_all_labels(), "SearchForm": SearchForm()})


def validation(request):
    return render(request, 'validation.html', {"Labels": get_all_labels(), "SearchForm": SearchForm()})


def login(request):
    return render(request, 'login.html', {"Labels": get_all_labels(), "SearchForm": SearchForm()})


def signup(request):
    return render(request, 'signup.html', {"Labels": get_all_labels(), "SearchForm": SearchForm()})


def blank_page(request):
    return render(request, 'blank-page.html', {"Labels": get_all_labels(), "SearchForm": SearchForm()})


def charts(request):
    return render(request, 'charts.html', {"Labels": get_all_labels(), "SearchForm": SearchForm()})


# 系统处理逻辑
def search(request):
    response = {"status": 0}
    print(request.POST)
    form = SearchForm(request.POST)
    if form.is_valid():
        response["label"] = request.POST["keyword"]
    else:
        response["status"] = -1
    return JsonResponse(response)


def update_image_label(request):
    """
    更新图片类别
    :param request: {oldURL: "战舰/zhanjian_01.jpg", newLabel: "坦克"}
    :return:
    """
    old_url = request.POST.get("oldURL").split("/")
    filename = old_url[1]
    old_label = old_url[0]
    new_label = request.POST.get("newLabel")
    old_url = osp.join(osp.join(osp.join(settings.BASE_DIR, "ImagesDB"), old_label), filename)
    new_url = osp.join(osp.join(osp.join(settings.BASE_DIR, "ImagesDB"), new_label), filename)
    if osp.exists(new_url):
        new_url = osp.splitext(new_url)[0] + ".1" + osp.splitext(new_url)[1]
    os.rename(old_url, new_url)
    response = {"status": 0}
    return JsonResponse(response)


def get_input_file_single(request):
    """
    获取上传的图片，该方法为按类别上传的处理逻辑
    :param request:
    :return:
    """
    response = {"status": 0}
    if request.method == "POST":
        label = request.POST.get("label")
        file_data = request.FILES.get(r"image_file", "没有图片")
        filename = file_data.name
        file_path = osp.join(osp.join(osp.join("/home/htc/Documents", "test"), label), filename)
        write2disk(file_path, file_data)
    return JsonResponse(response)


def get_input_file_multiple(request):
    """
    接收上传的.zip文件，保证文件类型正确
    :param request:
    :return:
    """
    response = {"status": 0}
    if request.method == "POST":
        file_data = request.FILES.get(r"image_file_zip", "没有数据")
        file_name = file_data.name
        file_path = osp.join(osp.join("/home/htc/Documents", "test"), file_name)
        write2disk(file_path, file_data)
        shutil.unpack_archive(file_path, osp.splitext(file_path)[0])
    return JsonResponse(response)


def delete_image(request):
    """
    删除图片
    :param request: 
    :return: 
    """
    response = {"status": 0}
    label, filename = resolve_file_name(request.POST.get("file_path"))
    file_path = osp.join(osp.join(osp.join(settings.BASE_DIR, "ImagesDB"), label), filename)
    print(file_path)
    # os.remove(file_path)
    return JsonResponse(response)


def get_classify_image(request):
    """
    保存待标注的上传图像
    :param request:
    :return:
    """
    response = {"status": 0}
    if request.method == "POST":
        file_data = request.FILES.get(r"image", "没有图片")
        filename = file_data.name
        file_path = osp.join(osp.join(settings.BASE_DIR, "media"), filename)
        write2disk(file_path, file_data)
    return JsonResponse(response)


def process_classify(request):
    """
    进行标注操作并返回结果
    :param request:
    :return: [{
                "name": 图片名称,
                "path": 图片路径,
                "probs": 前10个类的概率{
                    "类别1": 概率1, .....
                }
            },
                {},...]
    """
    labels = [item["label"] for item in get_all_labels()]
    result = []
    image_nums = 10
    image_names = np.random.randint(1, 11000, 100)
    for name in image_names:
        item = {}
        item["name"] = str(name)
        item["path"] = "/static/坦克/tank_1.jpg"
        item["probs"] = {}
        for label in labels:
            prob = round(np.random.random() * 100, 2)
            item["probs"][label] = prob
        result.append(item)
    content = render_to_string("results_cls.html", {"Results": result})
    return HttpResponse(content)


def process_classify_1_by_1(request):
    images = ["tank_1.jpg", "zhishengji_1.jpg"]
    labels = ["坦克", "直升机"]
    probs = [{"坦克": 97.35, "导弹": 2.79, "步枪": 1.90, "战斗机": 0.09, "潜水艇": 0.03},
             {"直升机": 87.23, "战斗机": 1.98, "雷达": 0.82, "手雷": 0.05, "手枪": 0.02}]
    result = []
    for idx, name in enumerate(images):
        prob = sorted(probs[idx].items(), key=lambda item: item[1], reverse=True)
        content = render_to_string("results_cls_v2.html",
                                   {"name": name, "path": "/static/{}/{}".format(labels[idx], name), "probs": prob[:5]})
        result.append(content)
    return HttpResponse("".join(result))
    # image = request.POST.get("image", "没有图片")
    # labels = [item["label"] for item in get_all_labels()]
    # probs = {label: round(np.random.random() * 100, 2) for label in labels}
    # probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    # content = render_to_string("results_cls_v2.html", {"name": image, "path": "/static/坦克/tank_10.jpg", "probs": probs[:5]})
    # return HttpResponse(content)


def process_train(request):
    print(request.POST)
    response = {"status": 0}
    form = TrainForm(request.POST)
    if form.is_valid():
        epochs = form.cleaned_data['epochs']
        batch_size = form.cleaned_data['batch_size']
        training_rate = form.cleaned_data['training_rate']
        decay_step = form.cleaned_data['decay_step']
        decay_rate = form.cleaned_data['decay_rate']
        layer = form.cleaned_data['layer']
        bottle_nodes = form.cleaned_data['bottle_nodes']
        param_initializer = form.cleaned_data['param_initializer']
        optimizer = form.cleaned_data['optimizer']
        classifier = form.cleaned_data['classifier']
        penalty = form.cleaned_data['penalty']
        penalty_param = form.cleaned_data['penalty_param']
        print("epochs: {}".format(epochs))
        print("batch_size: {}".format(batch_size))
        print("training_rate: {}".format(training_rate))
        print("decay_step: {}".format(decay_step))
        print("decay_rate: {}".format(decay_rate))
        print("layer: {}".format(layer))
        print("bottle_nodes: {}".format(bottle_nodes))
        print("param_initializer: {}".format(param_initializer))
        print("optimizer: {}".format(optimizer))
        print("classifier: {}".format(classifier))
        print("penalty: {}".format(penalty))
        print("penalty_param: {}".format(penalty_param))
    else:
        response["status"] = -1
        print("not valid")
    # time.sleep(10)
    return JsonResponse(response)


def process_validate(request):
    """
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

    Confusion confusion:
    [[87  0  0  0  1  0  0  0  0  0]
    [ 0 88  1  0  0  0  0  0  1  1]
    [ 0  0 85  1  0  0  0  0  0  0]
    [ 0  0  0 79  0  3  0  4  5  0]
    [ 0  0  0  0 88  0  0  0  0  4]
    [ 0  0  0  0  0 88  1  0  0  2]
    [ 0  1  0  0  0  0 90  0  0  0]
    [ 0  0  0  0  0  1  0 88  0  0]
    [ 0  0  0  0  0  0  0  0 88  0]
    [ 0  0  0  1  0  1  0  0  0 90]]
    :param request:
    :return:
    """
    response = {"status": 0, "details": ""}

    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    paths = [r"ImagesDB/战斗机/zhandouji_1.jpg", r"ImagesDB/雷达/leida_1.jpg",
             r"ImagesDB/战斗机/zhandouji_11.jpg", r"ImagesDB/战斗机/zhandouji_21.jpg",
             r"ImagesDB/雷达/leida_4.jpg", r"ImagesDB/潜艇/qianshuiting_1.jpg"]
    labels = ["雷达", "潜艇", "战斗机"]  # [item["label"] for item in get_all_labels()]
    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 1, 2, 2, 0, 2]
    report_str = classification_report(y_true=y_true, y_pred=y_pred, target_names=labels)
    print(report_str)
    matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
    for idx, item in enumerate(y_true):
        if item == y_pred[idx]:
            pass
        else:
            print("path: {}, y_true: {}, y_pred: {}".format(paths[idx], labels[item], labels[y_pred[idx]]))
            details = render_to_string("cls_errors.html",
                                       {"name": "/".join(paths[idx].split("/")[1:]),
                                        "path": "/static/" + "/".join(paths[idx].split("/")[1:]),
                                        "y_true": labels[item],
                                        "y_pred": labels[y_pred[idx]]})
            response["details"] += details

    report_str = '             precision    recall  f1-score   support\n\n          0       1.00      0.99      0.99        88\n          1       0.99      0.97      0.98        91\n          2       0.99      0.99      0.99        86\n          3       0.98      0.87      0.92        91\n          4       0.99      0.96      0.97        92\n          5       0.95      0.97      0.96        91\n          6       0.99      0.99      0.99        91\n          7       0.96      0.99      0.97        89\n          8       0.94      1.00      0.97        88\n          9       0.93      0.98      0.95        92\n\navg / total       0.97      0.97      0.97       899\n'
    report_matrix = [['', 'precision', 'recall', 'f1-score', 'support'],
                     ['0', '1.00', '0.99', '0.99', '88'],
                     ['1', '0.99', '0.97', '0.98', '91'],
                     ['2', '0.99', '0.99', '0.99', '86'],
                     ['3', '0.98', '0.87', '0.92', '91'],
                     ['4', '0.99', '0.96', '0.97', '92'],
                     ['5', '0.95', '0.97', '0.96', '91'],
                     ['6', '0.99', '0.99', '0.99', '91'],
                     ['7', '0.96', '0.99', '0.97', '89'],
                     ['8', '0.94', '1.00', '0.97', '88'],
                     ['9', '0.93', '0.98', '0.95', '92'],
                     ['total', '0.97', '0.97', '0.97', '899']]
    report = resolve_report(report_str)

    matrix = [[87, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 88, 1, 0, 0, 0, 0, 0, 1, 1],
              [0, 0, 85, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 79, 0, 3, 0, 4, 5, 0],
              [0, 0, 0, 0, 88, 0, 0, 0, 0, 4],
              [0, 0, 0, 0, 0, 88, 1, 0, 0, 2],
              [0, 1, 0, 0, 0, 0, 90, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 88, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 88, 0],
              [0, 0, 0, 1, 0, 1, 0, 0, 0, 90]]
    labels = [item["label"] for item in get_all_labels()]
    labels.insert(0, "")
    matrix.insert(0, labels)
    for idx, item in enumerate(matrix[1:]):
        item.insert(0, labels[idx + 1])

    idx = render_to_string("results_val.html", {"data": report, "is_matrix": False})
    confusion = render_to_string("results_val.html", {"data": matrix, "is_matrix": True})
    response["index"] = idx
    response["confusion"] = confusion
    return JsonResponse(response)
