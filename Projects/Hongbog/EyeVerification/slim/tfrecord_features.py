import tensorflow as tf

def int64_feature(value):
    """
    Wrapper for inserting int64 features into Example proto.
    :param value: instance for inserting int64 features into Example proto.
    :return: protocol buffer that is commonly used as data format which is int64 features
             for training and evaluation
    """
    if not isinstance(value, list): # value: instance, list: class name.
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def float_feature(value):
    """
    Wrapper for inserting float features into Example proto.
    :param value: instance for inserting float features into Example proto
    :return: protocol buffer that is commonly used as data format which is float features
             for training and evaluation
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def bytes_feature(value):
    """
    Wrapper for inserting bytes features into Example proto.
    :param value: instance for inserting bytes features into Example proto
    :return: protocol buffer that is commonly used as data format which is bytes features
             for training and evaluation
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))