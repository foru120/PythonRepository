※ 데이터 셋
 - CIFAR-10
  - class: 10
  - image per class: 50,000 (train), 10,000 (test)
  - size: 32 x 32
  - label: 0-airplane, 1-automobile, 2-bird, 3-cat, 4-deer, 5-dog, 6-frog, 7-horse, 8-ship, 9-truck

※ CIFAR-10 데이터 셋을 테스트 하기 위한 신경망
   (Weakly Supervised Medical Diagnosis and Localization from Multiple Resolutions)

▣ 특이사항


▣ 수행 로그
============================================= NasNet-A =============================================
   > LOG 1.
    - nasnet-A 모델 테스트(normal cell: x1, x2, x3, x4, x5, reduction cell: x1, x4, x5)
    - dataset: original cifar10 사용
    - optimizer learning rate decay: tf.train.cosine_decay
    - batch size: 32
    - Train: total epoch(149), max => epoch(146), acc(0.992), loss(0.267)
    - Validation: max => epoch(143), acc(0.885), loss(0.938)

   > LOG 2.
    - nasnet-A 모델 테스트(normal cell: prev_x, x1, x2, x3, x4, x5, reduction cell: x1, x2, x4, x5)
    - dataset: original cifar10 사용
    - optimizer learning rate decay: tf.train.cosine_decay
    - batch size: 32
    - Train: total epoch(80), max => epoch(75), acc(0.982), loss(0.452)
    - Validation: max => epoch(68), acc(0.863), loss(1.249)

   > LOG 3.
    - nasnet-A 모델 테스트(normal cell: prev_x, x1, x2, x3, x4, x5, reduction cell: x1, x2, x4, x5)
    - dataset: original cifar10 사용
    - optimizer learning rate decay: tf.train.cosine_decay, RMSProp Optimizer
    - batch size: 32
    - Train: total epoch(291), max => epoch(269), acc(1), loss(0.109)
    - Validation: max => epoch(262), acc(0.912), loss(0.707)

   > LOG 4.
    - nasnet-A 모델 테스트(normal cell: prev_x, x1, x2, x3, x4, x5, reduction cell: x1, x2, x4, x5)
    - LOG 3 까지 Normal Cell 과 Reduction Cell 의 출력이 잘못되어있어서 수정
    - dataset: original cifar10 사용
    - optimizer learning rate decay: tf.train.cosine_decay, RMSProp Optimizer
    - batch size: 32
    - Train: total epoch(), max => epoch(), acc(), loss()
    - Validation: max => epoch(), acc(), loss()