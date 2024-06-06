#!/usr/bin/env python
# coding: utf-8
# generate by : jupyter nbconvert --to  python 9_13_kaggle_dog_oc.ipynb

"""# 9.13 实战Kaggle比赛：狗的品种识别（ImageNet Dogs）
我们将在本节动手实战Kaggle比赛中的狗的品种识别问题。该比赛的网页地址是 https://www.kaggle.com/c/dog-breed-identification 。
在这个比赛中，将识别120类不同品种的狗。这个比赛的数据集实际上是著名的ImageNet的子集数据集。和上一节的CIFAR-10数据集中的图像不同，ImageNet数据集中的图像更高更宽，且尺寸不一。
图9.17展示了该比赛的网页信息。为了便于提交结果，请先在Kaggle网站上注册账号。
![狗的品种识别比赛的网页信息。比赛数据集可通过点击“Data”标签获取](http://zh.d2l.ai/_images/kaggle-dog.png)
首先，导入比赛所需的包或模块。
"""
# Install TensorFlow
# try:
#   # %tensorflow_version only exists in Colab.
#   %tensorflow_version 2.x
# #   get_ipython().run_line_magic('tensorflow_version', '2.x')
# except Exception:
#     pass
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import collections
import math
import random
class  Classifier():
    def __init__(self) -> None:
        self.train_ds= None
        self.valid_ds= None
        self.callback= None
        self.model = None
        self.train_valid_ds = None
        self.test_ds = None
        self.data_dir = '' # 数据集主目录
        self.idx_label = None # 标签字典
        self.lr = None
        self.batch_size = 0 # 批次大小

    def init_data(self, demo=True):
        """
        # ## 9.13.1 获取和整理数据集
# 比赛数据分为训练集和测试集。训练集包含了10,222张图像，测试集包含了10,357张图像。
两个数据集中的图像格式都是JPEG。这些图像都含有RGB三个通道（彩色），高和宽的大小不一。
训练集中狗的类别共有120种，如拉布拉多、贵宾、腊肠、萨摩耶、哈士奇、吉娃娃和约克夏等。
# """
        # 如果使用下载的Kaggle比赛的完整数据集，把demo变量改为False
        import zipfile
        self.data_dir = '../../data/kaggle_dog'
        if demo:
            zipfiles = ['train_valid_test_tiny.zip']
        else:
            zipfiles = ['train.zip', 'test.zip', 'labels.csv.zip']
        for f in zipfiles:
            with zipfile.ZipFile(self.data_dir + '/' + f, 'r') as z:
                z.extractall(self.data_dir)
    def  prepare_data(self, demo= True):
        self.init_data(demo)

        """
        # 因为我们在这里使用了小数据集，所以将批量大小设为1。
        在实际训练和测试时，我们应使用Kaggle比赛的完整数据集并调用`reorg_dog_data`函数来整理数据集。
        相应地，我们也需要将批量大小`batch_size`设为一个较大的整数，如128。
        """
        if demo:
            # 注意，此处使用小数据集并将批量大小相应设小。使用Kaggle比赛的完整数据集时可设批量大小
            # 为较大整数
            input_dir, self.batch_size = 'train_valid_test_tiny', 1
        else:
            label_file, train_dir, test_dir = 'labels.csv', 'train', 'test'
            input_dir, self.batch_size, valid_ratio = 'train_valid_test', 128, 0.1
            self.reorg_dog_data(self.data_dir, label_file, train_dir, test_dir, input_dir,
                        valid_ratio)
        self.load_data()

    def defind_model(self):
        """9.13.4 定义模型
        """
        from tensorflow.keras.applications import ResNet50
        net=ResNet50(
            input_shape=(224, 224, 3),
            weights='imagenet',
            include_top=False
        )
        self.model = tf.keras.Sequential([
            net,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(len(label_names), activation='softmax',dtype=tf.float32)
        ])
        self.model.summary()

    
    def reorg_train_valid(data_dir, train_dir, input_dir, valid_ratio, idx_label):
        """
        9.13.1.2 整理数据集
        我们定义下面的`reorg_train_valid`函数来从Kaggle比赛的完整原始训练集中切分出验证集。
        该函数中的参数`valid_ratio`指验证集中每类狗的样本数与原始训练集中数量最少一类的狗的样本数（66）之比。
        经过整理后，同一类狗的图像将被放在同一个文件夹下，便于稍后读取。
        """
        # 训练集中数量最少一类的狗的样本数
        min_n_train_per_label = (
            collections.Counter(idx_label.values()).most_common()[:-2:-1][0][1])
        # 验证集中每类狗的样本数
        n_valid_per_label = math.floor(min_n_train_per_label * valid_ratio)
        label_count = {}
        for train_file in os.listdir(os.path.join(data_dir, train_dir)): # data_dir,train_dir  = '../../data/kaggle_dog' ;'train'
            idx = train_file.split('.')[0]
            label = idx_label[idx]
            d2l.mkdir_if_not_exist([data_dir, input_dir, 'train_valid', label])
            shutil.copy(os.path.join(data_dir, train_dir, train_file),
                        os.path.join(data_dir, input_dir, 'train_valid', label))
            if label not in label_count or label_count[label] < n_valid_per_label:
                d2l.mkdir_if_not_exist([data_dir, input_dir, 'valid', label])
                shutil.copy(os.path.join(data_dir, train_dir, train_file),
                            os.path.join(data_dir, input_dir, 'valid', label))
                label_count[label] = label_count.get(label, 0) + 1
            else:
                d2l.mkdir_if_not_exist([data_dir, input_dir, 'train', label])
                shutil.copy(os.path.join(data_dir, train_dir, train_file),
                            os.path.join(data_dir, input_dir, 'train', label))


    def reorg_dog_data(self,data_dir, label_file, train_dir, test_dir, input_dir,
                    valid_ratio):
        """
        读取训练数据标签、切分验证集并整理测试集。
        """
        # 读取训练数据标签
        with open(os.path.join(data_dir, label_file), 'r') as f:
            # 跳过文件头行（栏名称）
            lines = f.readlines()[1:]
            tokens = [l.rstrip().split(',') for l in lines]
            self.idx_label = dict(((idx, label) for idx, label in tokens))
        self.reorg_train_valid(data_dir, train_dir, input_dir, valid_ratio, self.idx_label)
        # 整理测试集
        d2l.mkdir_if_not_exist([data_dir, input_dir, 'test', 'unknown'])
        for test_file in os.listdir(os.path.join(data_dir, test_dir)):
            shutil.copy(os.path.join(data_dir, test_dir, test_file),
                        os.path.join(data_dir, input_dir, 'test', 'unknown'))


    def transform_train(self,imgpath,label):
        """
        9.13.2 图像增广
        本节比赛的图像尺寸比上一节中的更大。这里列举了更多可能有用的图像增广操作。
        """
        # 随机对图像裁剪出面积为原图像面积0.08~1倍、且高和宽之比在3/4~4/3的图像，再放缩为高和
        # 宽均为224像素的新图像
        feature=tf.io.read_file(imgpath)
        feature = tf.image.decode_jpeg(feature,channels=3)
        feature = tf.image.resize(feature, size=[400, 400])
        seed=random.randint(8,100)/100
        feature = tf.image.random_crop(feature, size=[int(seed*feature.shape[0]), int(seed*feature.shape[1]), 3])
        feature = tf.image.resize(feature, size=[224, 224])
        feature = tf.image.random_flip_left_right(feature)
        feature = tf.image.random_flip_up_down(feature)
        # 标准化
        feature = tf.divide(feature, 255.)
        # 正则化
        mean = tf.convert_to_tensor([0.485, 0.456, 0.406])
        std = tf.convert_to_tensor([0.229, 0.224, 0.225])
        feature = tf.divide(tf.subtract(feature, mean), std)
        #feature = tf.image.per_image_standardization(feature)
        #print(feature,label)
        return tf.image.convert_image_dtype(feature, tf.float32),label



    def transform_test(self,imgpath,label):
        # 测试时，我们只使用确定性的图像预处理操作。
        feature=tf.io.read_file(imgpath)
        feature = tf.image.decode_jpeg(feature,channels=3)
        feature = tf.image.resize(feature, [224, 224])
        feature = tf.divide(feature, 255.)
        # feature = tf.image.per_image_standardization(feature)
        mean = tf.convert_to_tensor([0.485, 0.456, 0.406])
        std = tf.convert_to_tensor([0.229, 0.224, 0.225])
        feature = tf.divide(tf.subtract(feature, mean), std)
        return feature,label

    def load_data(self):
        """9.13.3 读取数据集
        获取所有图片path和label"""
        import pathlib
        data_root="../../data/kaggle_dog/train_valid_test_tiny"
        train_data_root = pathlib.Path(data_root+"/train")
        valid_data_root = pathlib.Path(data_root+"/valid")
        train_valid_data_root = pathlib.Path(data_root+"/train_valid")
        test_data_root = pathlib.Path(data_root+"/test")
        label_names = sorted(item.name for item in train_data_root.glob('*/') if item.is_dir())
        label_to_index = dict((name, index) for index, name in enumerate(label_names)) # lable to index relationship

        train_all_image_paths = [str(path) for path in list(train_data_root.glob('*/*'))]
        valid_all_image_paths = [str(path) for path in list(valid_data_root.glob('*/*'))]
        train_valid_all_image_paths = [str(path) for path in list(train_valid_data_root.glob('*/*'))]
        test_all_image_paths = [str(path) for path in list(test_data_root.glob('*/*'))]


        train_all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in train_all_image_paths]
        valid_all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in valid_all_image_paths]
        train_valid_all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in train_valid_all_image_paths]
        test_all_image_labels = [-1 for i in range(len(test_all_image_paths))]
        print("First 10 images indices: ", train_valid_all_image_labels[:10])
        print("First 10 labels indices: ", train_valid_all_image_labels[:10])


        # 构建一个 tf.data.Dataset
        # 图像加强

        self.train_ds = tf.data.Dataset.from_tensor_slices((train_all_image_paths, train_all_image_labels)).\
            map(self.transform_train).shuffle(len(train_all_image_paths)).batch(self.batch_size)
        self.valid_ds = tf.data.Dataset.from_tensor_slices((valid_all_image_paths, valid_all_image_labels)).\
            map(self.transform_train).shuffle(len(valid_all_image_paths)).batch(self.batch_size)
        train_valid_ds = tf.data.Dataset.from_tensor_slices((train_valid_all_image_paths, train_valid_all_image_labels)).\
            map(self.transform_train).shuffle(len(train_valid_all_image_paths)).batch(self.batch_size)
        self.test_ds = tf.data.Dataset.from_tensor_slices((test_all_image_paths, test_all_image_labels)).\
            map(self.transform_test).shuffle(len(test_all_image_paths)).batch(self.batch_size)


    def defind_train_func(self):
        # #### 9.13.5. 定义训练函数
        self.lr = 0.1
        lr_decay = 0.01

        def scheduler(epoch):
            if epoch < 10:
                return lr
            else:
                return lr * tf.math.exp(lr_decay * (10 - epoch))

        self.callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

        self.model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr, momentum=0.9),
                loss='sparse_categorical_crossentropy')
        
    def train_model(self):
        """# #### 9.13.6. 训练模型"""
        self.model.fit(self.train_ds, epochs=1 , validation_data=self.valid_ds,  callbacks=[self.callback])

    


        # ### 9.13.7 对测试集分类并在Kaggle提交结果
        # 得到一组满意的模型设计和超参数后，我们使用全部训练数据集（含验证集）重新训练模型，
        # 并对测试集分类。注意，我们要用刚训练好的输出网络做预测。
        self.model.compile(optimizer=keras.optimizers.SGD(learning_rate=self.lr, momentum=0.9),
                loss='sparse_categorical_crossentropy')
        self.model.fit(self.train_valid_ds, epochs=1 , callbacks=[self.callback])

    def predict(self):
        probabilities=self.model.predict(self.test_ds)
        predictions=np.argmax(probabilities, axis=-1)
        self.write_ret(predictions)
    
    def write_ret(self,predictions):

        ids = sorted(os.listdir(os.path.join(data_dir, input_dir, 'test/unknown')))
        with open('submission.csv', 'w') as f:
            f.write('id,' + "preds"+ '\n')
            for i, output in zip(ids, predictions):
                f.write(i.split('.')[0] + ',' + str(output) + '\n')


# #### 小结
# 
# * 我们可以使用在ImageNet数据集上预训练的模型抽取特征，并仅训练自定义的小规模输出网络，从而以较小的计算和存储开销对ImageNet的子集数据集做分类。

# 1. 32中predictions 的ouput 13 代表什么类别
# 2. 预测的具体步骤是那些
# 3. 如何通过加入一个新的图片,并进行预测

if __name__ == '__main__':
    c = Classifier()
    # 数据准备相关
    c.prepare_data()
    # raise Exception

    # 模型参数相关
    c.defind_model()
    c.defined_train_func()

    # 训练相关
    c.train_model()

    # 预测
    c.predict()


