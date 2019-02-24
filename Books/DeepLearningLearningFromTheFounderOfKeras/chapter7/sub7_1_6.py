#todo p.328 ~ p.329
#todo code x ~ code x
#todo 7.1.6 층과 모델

from keras import layers
from keras import applications
from keras import Input

xception_base = applications.Xception(weights=None,
                                      include_top=False)
left_input = Input(shape=(250, 250, 3))
right_input = Input(shape=(250, 250, 3))

left_features = xception_base(left_input)
right_features = xception_base(right_input)

merged_features = layers.concatenate(inputs=[left_features, right_features], axis=-1)