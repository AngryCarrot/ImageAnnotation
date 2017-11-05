from django import forms


class TrainForm(forms.Form):
    epochs = forms.IntegerField(label="Epochs",
                                initial=1000,
                                widget=forms.TextInput(attrs={"class": "form-control1",
                                                              "placeholder": "训练轮数"
                                                              }),
                                help_text="如：1500"
                                )
    batch_size = forms.IntegerField(label="Batch Size",
                                    initial=32,
                                    widget=forms.TextInput(attrs={"class": "form-control1",
                                                                  "placeholder": "每批次训练图像数量"
                                                                  }),
                                    help_text="如：32"
                                    )
    training_rate = forms.FloatField(label="Training Rate",
                                     initial=0.001,
                                     widget=forms.TextInput(attrs={"class": "form-control1",
                                                                   "placeholder": "学习率"
                                                                   }),
                                     help_text="如：0.001"
                                     )
    decay_step = forms.IntegerField(label="Decay Step",
                                    initial=2000,
                                    widget=forms.TextInput(attrs={"class": "form-control1",
                                                                  "placeholder": "学习率衰减时间"
                                                                  }),
                                    help_text="如：5000"
                                    )
    decay_rate = forms.FloatField(label="Decay Rate",
                                  initial=0.9,
                                  widget=forms.TextInput(attrs={"class": "form-control1",
                                                                "placeholder": "学习率衰减大小"
                                                                }),
                                  help_text="如：0.9"
                                  )
    layer = forms.ChoiceField(label="Feature Layer",
                              widget=forms.Select(attrs={"class": "form-control1",
                                                         "placeholder": ""
                                                         }),
                              choices=(("pool5", "pool5"), ("fc6", "fc6"), ("fc6", "fc6"),
                                       ("fc7", "fc7"), ("prob", "prob")),
                              help_text="网络某层结果"
                              )
    bottle_nodes = forms.IntegerField(label="Bottle Nodes",
                                      initial=1000,
                                      widget=forms.TextInput(attrs={"class": "form-control1",
                                                                    "placeholder": "自编码器编码层维度"
                                                                    }),
                                      help_text="如：1000"
                                      )
    param_initializer = forms.ChoiceField(label="Parameter Initializer",
                                          widget=forms.Select(attrs={"class": "form-control1",
                                                                     "placeholder": ""
                                                                     }),
                                          choices=(("Random Normal Distribution", "Random Normal Distribution"), ("Xarvier", "Xarvier"), ("Uniform Distribution", "Uniform Distribution"), ("Zeros", "Zeros"),
                                                   ("Ones", "Ones")),
                                          help_text="权重初始化方式"
                                          )
    optimizer = forms.ChoiceField(label="Optimizer",
                                  widget=forms.Select(attrs={"class": "form-control1",
                                                             "placeholder": ""
                                                             }),
                                  choices=(
                                      ("GradientDescentOptimizer", "GradientDescentOptimizer"), ("RMSPropOptimizer", "RMSPropOptimizer"),
                                      ("AdamOptimizer", "AdamOptimizer")),
                                  help_text="参数求解算法"
                                  )
    classifier = forms.ChoiceField(label="Classifier",
                                   widget=forms.Select(attrs={"class": "form-control1",
                                                              "placeholder": ""
                                                              }),
                                   choices=(
                                       ("SVM", "SVM"),
                                       ("LogisticRegression", "LogisticRegression")),
                                   help_text="分类器类型"
                                   )
    penalty = forms.ChoiceField(label="Penalty",
                                widget=forms.Select(attrs={"class": "form-control1",
                                                           "placeholder": ""
                                                           }),
                                choices=(
                                    ("L1", "L1"),
                                    ("L2", "L2"),
                                    ("ElasticNet", "ElasticNet")),
                                help_text="分类器正则化项"
                                )
    penalty_param = forms.FloatField(label="Penalty Param",
                                     initial=0.1,
                                     widget=forms.TextInput(attrs={"class": "form-control1",
                                                                   "placeholder": "正则化参数"
                                                                   }),
                                     help_text="如：0.01"
                                     )
