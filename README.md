Xception详解
===========

# 一. 简介

> Xception是google继Inception后提出的对Inception v3的另一种改进，主要是采用depthwise separable convolution来替换原来Inception v3中的卷积操作。depthwise separable convolution的结构类似Mobile Net，可参考：https://yinguobing.com/separable-convolution/. 。

# 二. 网络结构

> Xception结构如图所示：



# 三. 代码

> 利用MNIST数据集，构建MobileNets网络，查看网络效果，由于输入为28 x 28，所以最后的全局池化没有用到，代码如下：

```python

```
