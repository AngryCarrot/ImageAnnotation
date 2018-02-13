"""ImageAnnotation URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.10/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin

from Annotation import views

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    # Views html，原本的HTML框架页面
    url(r'^origin/dashboard/$', views.dashboard, name='dashboard'),
    url(r'^origin/grids/$', views.grids, name='grids'),
    url(r'^origin/media/$', views.media, name='media'),
    url(r'^origin/general/$', views.general, name='general'),
    url(r'^origin/typography/$', views.typography, name='typography'),
    url(r'^origin/widgets/$', views.widgets, name='widgets'),
    url(r'^origin/inbox/$', views.inbox, name='inbox'),
    url(r'^origin/compose/$', views.compose, name='compose'),
    url(r'^origin/tables/$', views.tables, name='tables'),
    url(r'^origin/forms/$', views.forms, name='forms'),
    url(r'^origin/validation/$', views.validation, name='validation'),
    url(r'^origin/login/$', views.login, name='login'),
    url(r'^origin/signup/$', views.signup, name='signup'),
    url(r'^origin/blank_page/$', views.blank_page, name='blank-page'),
    url(r'^origin/charts/$', views.charts, name='charts'),

    # 有效页面
    # 首页
    url(r'^$', views.index, name='index'),
    url(r'^index/$', views.index, name='index'),
    # 图库页面
    url(r'^gallery$', views.gallery2, name='gallery_no_params'),
    url(r'^gallery/(\w+)$', views.gallery, name='gallery'),
    # 图库幻灯片浏览
    url(r'^slider/$', views.slider, name='slider'),
    # 数据集各类别浏览
    url(r'^overview/$', views.overview, name='overview'),
    # 上传，单一类别上传和完整类别上传由后面的参数进行区分
    url(r'^upload/(\w+)/$', views.upload, name='upload'),
    # 标注处理
    url(r'^classify/$', views.classify, name='classify'),
    # 训练
    url(r'^train/$', views.train, name='train'),
    # 模型验证
    url(r'^validate/$', views.validate, name='validate'),
    # 搜索
    url(r'^search/$', views.search, name='search'),

    # 图片处理过程
    # 更改类别
    url(r'^update/label/$', views.update_image_label, name='update_label'),
    # 单一类别上传
    url(r'^upload/single/process/$', views.get_input_file_single, name='upload_single'),
    # 多类别上传
    url(r'^upload/multiple/process/$', views.get_input_file_multiple, name='upload_multiple'),
    # 删除图片
    url(r'^delete/$', views.delete_image, name="delete_image"),

    # 标注相关过程
    url(r'^upload/classify/images/$', views.get_classify_image, name="upload_classify"),
    url(r"^classify/process/$", views.process_classify, name="process_classify"),
    # url(r"^classify/process/$", views.classify_result, name="process_classify"), # 跳转到新页面，已废弃
    url(r'^classify/process/1_by_1$', views.process_classify_1_by_1, name="process_classify_1_by_1"),

    # 训练过程
    url(r'^train/process/$', views.process_train, name="process_train"),

    # 验证结果
    url(r"^validate/process/$", views.process_validate, name="process_validate"),
]
