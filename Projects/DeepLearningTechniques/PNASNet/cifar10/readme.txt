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
============================================= PNASNet-5 =============================================
 > LOG 1.
  - PNASNet-5 cifar10 데이터 셋 기반 신경망 모델
  - Train: epoch(), acc(), loss(), time()
  - Validation: acc(), loss()
  - Test: acc(), loss()