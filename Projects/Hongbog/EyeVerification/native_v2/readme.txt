Tensor("right/input_module/low_res_X:0", shape=(?, 60, 160, 1), dtype=float32)
Tensor("right/input_module/mid_res_X:0", shape=(?, 80, 200, 1), dtype=float32)
Tensor("right/input_module/high_res_X:0", shape=(?, 100, 240, 1), dtype=float32)
Tensor("left/input_module/low_res_X:0", shape=(?, 60, 160, 1), dtype=float32)
Tensor("left/input_module/mid_res_X:0", shape=(?, 80, 200, 1), dtype=float32)
Tensor("left/input_module/high_res_X:0", shape=(?, 100, 240, 1), dtype=float32)

Tensor("right/output_module/softmax:0", shape=(?, 7), dtype=float32)
Tensor("left/output_module/softmax:0", shape=(?, 7), dtype=float32)

▣ 001
 ■ Data Augmentation
  - Random Crop
  - Random Brightness
  - Random contrast
  - Random Hue
  - Random Saturation

 ■ Model
  - Multi-scale + MobileNet-v2

 ■ Parameters
  - Epoch: 20
  - Dropout Rate: 0.4
  - Batch Size: 50
  - Learning Rate: 0.001
  - Regularization Scale: 0.0005

  - Batch Normalization
   - decay: 0.9
   - center=True
   - scale=True
   - fused=True
   - updates_collections=tf.GraphKeys.UPDATE_OPS

▣ 002
 - tf.image.resize_nearest_neighbor -> tf.layers.conv2d_transpose 로 변경
 - 나머지 구성은 001 과 동일

▣ 003
 - Data Augmentation 에서 equalization 추가
 - 나머지 구성은 002 과 동일

▣ 004
 - eye_dataset_v2 로 학습
 - scale -> (60, 160), (80, 200), (100, 240)

▣ 005
 - equalization
 - eye_dataset_v2 로 학습
 - scale -> (46, 100), (70, 150), (92, 200)

▣ 006
 - no equalization
 - eye_dataset_v1 학습
 - tf.image.random_brightness -> 80.
 - scale -> (46, 100), (70, 150), (92, 200)
 - epoch: 20
 - accuracy: 87%

▣ 007
 - no equalization
 - eye_dataset_v1 학습
 - eye_dataset_v2 학습
 - tf.image.random_brightness -> 80.
 - scale -> (46, 100), (70, 150), (92, 200)
 - accuracy: 85%

▣ 008
 - equalization
 - eye_dataset_v1 학습
 - tf.image.random_brightness -> 80.
 - scale -> (46, 100), (70, 150), (92, 200)
 - accuracy: 86%

▣ 009
 - equalization
 - eye_dataset_v1 학습
 - eye_dataset_v2 학습
 - tf.image.random_brightness -> 80.
 - scale -> (46, 100), (70, 150), (92, 200)
 - accuracy: 85%

▣ 010
 - equalization
 - tf.image.random_brightness -> 80.
 - eye_dataset_v2 학습
 - scale -> (46, 100), (70, 150), (92, 200)
 - accuracy: 85%

▣ 011
 - no equalization
 - eye_dataset_v2 학습
 - scale -> (46, 100), (70, 150), (92, 200)
 - accuracy: 99.9%

▣ 012
 - equalization
 - eye_dataset_v2 학습
 - scale -> (46, 100), (70, 150), (92, 200)
 - accuracy: 99.9%