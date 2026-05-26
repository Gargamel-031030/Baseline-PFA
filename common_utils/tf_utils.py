from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import numpy as np
np.random.seed(10)


def global_step_creator():
    global_step = [v for v in tf.global_variables() if v.name == "global_step:0"][0]
    global_step_placeholder = tf.placeholder(dtype=tf.float32, 
                                            shape=(), 
                                            name='global_step_placeholder')
    set_global_step = tf.assign(global_step, global_step_placeholder)
    return set_global_step
    '''
    one = tf.constant(1, dtype=tf.float32, name='one')
    new_global_step = tf.add(global_step, one)
    increase_global_step = tf.assign(global_step, new_global_step)
    '''

## 生成一组 TensorFlow 赋值操作，将传入的字典 dic 中的值赋给相应的可训练变量。
## 使用 tf.assign 将字典中每个变量的值赋给对应的 TensorFlow 变量。
def Assignements(dic):
    return [tf.assign(var, dic[Vname_to_Pname(var)]) for var in tf.trainable_variables()]

## 将变量名称转换为占位符名称。
## 通过去除变量名称中的 : 以及后面的部分，添加 _placeholder 后缀来生成占位符名称。
## 例如，weight:0 映射为 weight_placeholder
def Vname_to_Pname(var):
    return var.name[:var.name.find(':')] + '_placeholder'

## 将变量名称转换为 TensorFlow feed_dict 使用的占位符名称。
## 通过去除变量名称中的 : 以及后面的部分，添加 _placeholder:0 后缀来生成占位符名称。
## 例如，weight:0 映射为 weight_placeholder:0
def Vname_to_FeedPname(var):
    return var.name[:var.name.find(':')] + '_placeholder:0'

## 去掉变量名称中的 : 以及后面的部分，获取纯变量名称。
## 例如，weight:0 映射为 weight
def Vname_to_Vname(var):
    return var.name[:var.name.find(':')]
