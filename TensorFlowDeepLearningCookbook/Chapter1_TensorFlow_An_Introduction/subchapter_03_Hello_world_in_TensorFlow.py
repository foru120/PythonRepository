import tensorflow as tf  # tensorflow 임포트
message = tf.constant('Welcome to the exciting world of Deep Neural Networks!')  # 상수 문자열을 사용하기 위해 tf.constant 사용
with tf.Session() as sess:  # graph element 를 실행하기 위해 with 절을 사용해 Session 을 정의
    print(sess.run(message).decode())  # run 메서드를 사용해 session 수행, decode 함수 미사용시 byte 형태로 출력

'''
    ▣ warning message and information message 없애는 방법
     level 1: information message
     level 2: warning message
     level 3: error message
'''
import os
# os.environ('TF_CPP_MIN_LOG_LEVEL')='2'  # level 2 까지 모든 메세지를 무시