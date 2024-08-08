# import  os

# print(os.listdir(os.path.dirname(os.path.abspath(__file__))))
# import pathlib
# data_root="../../data/kaggle_dog/train_valid_test_tiny"
# train_data_root = pathlib.Path(data_root+"/train")
# # path = '../../data/kaggle_dog/train_valid_test_tiny/train'
# print(train_data_root, type(train_data_root))
# # print(train_data_root.glob('*/'))
# label_names = sorted(item.name for item in train_data_root.glob('*/') if item.is_dir())
# # print(label_names)
# train_all_image_paths = [str(path) for path in list(train_data_root.glob('*/*'))]

# label_to_index = dict((name, index) for index, name in enumerate(label_names)) # lable to index relationship
# print(label_to_index)

# 梯度下降案例
# import tensorflow as tf
# opt = tf.keras.optimizers.SGD(learning_rate=0.1)
# var = tf.Variable(1.0)
# print(var)
# loss = lambda: (var ** 2)/2.0         # d(loss)/d(var1) = var1
# step_count = opt.minimize(loss, [var]).numpy()
# print(step_count)
# # Step is `-learning_rate*grad`
# var.numpy()
# print(var.numpy())

# demo 2
# opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
# var = tf.Variable(1.0)
# val0 = var.value()
# loss = lambda: (var ** 2)/2.0         # d(loss)/d(var1) = var1
# # First step is `-learning_rate*grad`
# step_count = opt.minimize(loss, [var]).numpy()
# val1 = var.value()
# print(val0 , val1)
# (val0 - val1).numpy()
def  main(print_type= ''):
    import  tensorflow as  tf
    if print_type == 'layers':
        from tensorflow.keras.layers import Input

        print('>>>')
        # 创建输入层
        try:

            input_layer = Input(shape=(224, 224, 3))

            # 获取输入层的形状
            input_shape = input_layer.shape

            # 打印输入层的形状
            print("Input Layer Shape:", input_shape)

            # 打印输入层的详细信息
            print("Input Layer Details:")
            print(input_layer)
            """
            Tensor("input_1:0", shape=(None, 224, 224, 3), dtype=float32)
            输入层的名称和索引
            """
        except Exception as e:
            print(str(e))
    elif print_type == 'layers_model':
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Subtract

        # 定义输入层
        input_1 = Input(shape=(224, 224, 3))
        input_2 = Input(shape=(224, 224, 3))

        # 定义减法层
        output = Subtract()([input_1, input_2])

        # 创建模型
        model = Model(inputs=[input_1, input_2], outputs=output)

        # 打印模型结构
        model.summary()
    else:
        print('hello world2')

if __name__ == '__main__':
    print_type = 'layers_model'
    # print_type = 'layers'
    main(print_type)


