Xception详解
===========

# 一. 简介

> Xception是google继Inception后提出的对Inception v3的另一种改进，主要是采用depthwise separable convolution来替换原来Inception v3中的卷积操作。depthwise separable convolution的结构类似Mobile Net，可参考：https://yinguobing.com/separable-convolution/. 。

# 二. 网络结构

> Xception结构如图所示：

![image](https://github.com/ShaoQiBNU/CV-Xception/blob/master/images/1.png)


# 三. 代码

> 利用MNIST数据集，构建MobileNets网络，查看网络效果，由于输入为28 x 28，所以最后的全局池化没有用到，代码如下：

```python
##################### load packages #####################
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data


##################### load data ##########################
mnist = input_data.read_data_sets("mnist_sets", one_hot=True)

##################### set net hyperparameters #####################
learning_rate = 0.01

epochs = 20
batch_size_train = 128
batch_size_test = 100

display_step = 20

########### set net parameters ##########
#### img shape:28*28 ####
n_input = 784

#### 0-9 digits ####
n_classes = 10

##################### placeholder #####################
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])


##################### build net model ##########################
############## depthwise separable_conv —— entry flow and exit flow ################
def depthwise_separable_conv1(inputs, filter, kernel_size, stride):

    layer1 = slim.separable_conv2d(inputs, num_outputs=filter, stride=stride, depth_multiplier=1, kernel_size=kernel_size)
    layer1 = slim.batch_norm(layer1)
    layer1 = tf.nn.relu(layer1)

    layer2 = slim.separable_conv2d(layer1, num_outputs=filter, stride=stride, depth_multiplier=1, kernel_size=kernel_size)
    layer2 = slim.batch_norm(layer2)
    layer2 = slim.max_pool2d(layer2, stride=2, kernel_size=kernel_size, padding='SAME')

    return layer2

############## depthwise separable_conv —— middle flow ################
def depthwise_separable_conv2(inputs, filter, kernel_size, stride):

    layer = tf.nn.relu(inputs)
    layer = slim.separable_conv2d(layer, num_outputs=filter, stride=stride, depth_multiplier=1, kernel_size=kernel_size)
    layer = slim.batch_norm(layer)

    return layer


##################### MobileNet #####################
def XceptionNet(x, n_classes):

    ############ reshape input picture #############
    x = tf.reshape(x, shape=[-1, 28, 28, 1])


    ############ Entry flow #############
    ###### first convolution ######
    net = slim.conv2d(x, num_outputs=32, kernel_size=[3, 3], stride=2, padding='SAME')
    net = slim.batch_norm(net)
    net = tf.nn.relu(net)

    ###### second convolution ######
    net = slim.conv2d(net, num_outputs=64, kernel_size=[3, 3], stride=1, padding='SAME')
    net = slim.batch_norm(net)
    net = tf.nn.relu(net)

    ###### depthwise separable convolution 1 ######
    ###### residual ######
    residual = slim.conv2d(net, num_outputs=128, kernel_size=[1, 1], stride=2, padding='SAME')
    residual = slim.batch_norm(residual)
    net = depthwise_separable_conv1(net, filter=128, kernel_size=[3, 3], stride=1)
    net = tf.add(net, residual)


    ###### depthwise separable convolution 2 ######
    ###### residual ######
    residual = slim.conv2d(net, num_outputs=256, kernel_size=[1, 1], stride=2, padding='SAME')
    residual = slim.batch_norm(residual)
    net = depthwise_separable_conv1(net, filter=256, kernel_size=[3, 3], stride=2)
    net = tf.add(net, residual)


    ###### depthwise separable convolution 3 ######
    ###### residual ######
    residual = slim.conv2d(net, num_outputs=728, kernel_size=[1, 1], stride=2, padding='SAME')
    residual = slim.batch_norm(residual)
    net = depthwise_separable_conv1(net, filter=728, kernel_size=[3, 3], stride=2)
    net = tf.add(net, residual)


    ############ Middle flow #############
    for i in range(8):
        residual = net
        for j in range(3):
            net = depthwise_separable_conv2(net, filter=728, kernel_size=[3, 3], stride=1)

        net = tf.add(net, residual)


    ############ Exit flow #############

    ###### depthwise separable convolution ######
    ###### residual ######
    residual = slim.conv2d(net, num_outputs=1024, kernel_size=[1, 1], stride=2, padding='SAME')
    residual = slim.batch_norm(residual)

    net = depthwise_separable_conv2(net, filter=728, kernel_size=[3, 3], stride=2)
    net = depthwise_separable_conv2(net, filter=1024, kernel_size=[3, 3], stride=2)
    net = slim.max_pool2d(net, stride=2, kernel_size=[3, 3], padding='SAME')
    net = tf.add(net, residual)

    ###### depthwise separable convolution ######
    net = slim.separable_conv2d(net, num_outputs=1536, stride=1, depth_multiplier=1, kernel_size=[3, 3])
    net = slim.batch_norm(net)
    net = tf.nn.relu(net)

    ###### depthwise separable convolution ######
    net = slim.separable_conv2d(net, num_outputs=2048, stride=1, depth_multiplier=1, kernel_size=[3, 3])
    net = slim.batch_norm(net)
    net = tf.nn.relu(net)

    ####### 全局平均池化 ########
    # pool1 = slim.avg_pool2d(net, [10,10], stride=1)
    pool1 = slim.avg_pool2d(net, [1, 1], stride=1)

    ####### flatten 影像展平 ########
    flatten = tf.reshape(pool1, (-1, 1 * 1 * 2048))

    ####### out 输出，10类 可根据数据集进行调整 ########
    out = tf.layers.dense(flatten, n_classes)

    return out


##################### define model, loss and optimizer #####################
#### model pred 影像判断结果 ####
pred = XceptionNet(x, n_classes)

#### loss 损失计算 ####
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

#### optimization 优化 ####
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#### accuracy 准确率 ####
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


##################### train and evaluate model ##########################
########## initialize variables ##########
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1

    #### epoch 世代循环 ####
    for epoch in range(epochs + 1):

        #### iteration ####
        for _ in range(mnist.train.num_examples // batch_size_train):

            step += 1

            ##### get x,y #####
            batch_x, batch_y = mnist.train.next_batch(batch_size_train)

            ##### optimizer ####
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

            ##### show loss and acc #####
            if step % display_step == 0:
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
                print("Epoch " + str(epoch) + ", Minibatch Loss=" + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))

    print("Optimizer Finished!")

    ##### test accuracy #####
    for _ in range(mnist.test.num_examples // batch_size_test):
        batch_x, batch_y = mnist.test.next_batch(batch_size_test)
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: batch_x, y: batch_y}))
```


# 四. 参考

https://github.com/kwotsin/TensorFlow-Xception

https://blog.csdn.net/wangli0519/article/details/73863855
