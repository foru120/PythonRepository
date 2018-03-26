import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import ImageGrab
from drawnow import drawnow
from PIL import Image
import os

from DeepLearningTechniques.DC_GAN.dcgan_gogh import DCGAN

batch_size = 100

def read_data():
    tot_image_data = []
    for idx, filename in enumerate(os.listdir('gogh_data\\')):
        im = Image.open('D:\\05_source\\PythonRepository\\DeepLearningTechniques\\DC_GAN\\gogh_data\\' + str(filename))
        load_img = im.load()
        temp_data = []
        for i in range(0, 256):
            for j in range(0, 256):
                temp_data.append((np.array(load_img[i, j]) / 255).tolist())
        tot_image_data.append(temp_data)
        # if idx >= 99:
        #     break
    return np.reshape(np.array(tot_image_data), (-1, 256, 256, 3))

def image_screeshot():
    im = ImageGrab.grab()
    im.show()

# monitoring 관련 parameter
value_num = 2
mon_epoch_list = []
mon_value_list = [[] for _ in range(value_num)]
mon_color_list = ['blue', 'yellow', 'red', 'cyan', 'magenta', 'green', 'black']
mon_label_list = ['g_loss', 'd_loss']

def monitor_train_cost():
    for cost, color, label in zip(mon_value_list, mon_color_list[0:len(mon_label_list)], mon_label_list):
        plt.plot(mon_epoch_list, cost, c=color, lw=2, ls="--", marker="o", label=label)
    plt.title('DC-GAN Loss')
    plt.legend(loc=1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

# 모든 변수들의 값을 출력하는 함수
def get_model_params():
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

# 이전 상태를 restore 하는 함수
def restore_model_params(model_params):
    gvar_names = list(model_params.keys())
    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + '/Assign') for gvar_name in gvar_names}
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}  # inputs : 해당 operation 의 입력 데이터를 표현하는 objects
    feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)

def create_image(images, epoch):
    images = tf.image.convert_image_dtype(tf.div(tf.add(images[:10], 1.0), 2.0), tf.uint8)
    images = [image for image in tf.split(images, 10, axis=0)]

    for idx in range(0, len(images)):
        if not os.path.isdir('gogh_gen_image\\1th_test\\' + str(epoch)):
            os.mkdir('gogh_gen_image\\1th_test\\' + str(epoch))
            tf.write_file('gogh_gen_image\\1th_test\\' + str(epoch) + '\\' + str(idx) + '.jpeg', tf.image.encode_jpeg(tf.squeeze(images[idx], [0])))

    # for idx in range(0, len(images)):
    #     new_img = Image.new("RGB", (256, 256), "white")
    #     for i in range(0, new_img.size[0]):
    #         for j in range(0, new_img.size[1]):
    #             new_img.putpixel((i, j), (int(images[idx][i][j][0]), int(images[idx][i][j][1]), int(images[idx][i][j][2])))
    #
    #     if not os.path.isdir('gogh_gen_image\\1th_test\\' + str(epoch)):
    #         os.mkdir('gogh_gen_image\\1th_test\\' + str(epoch))
    #
    #     new_img.save('gogh_gen_image\\1th_test\\' + str(epoch) + '\\' + str(idx) + '.jpeg')

def save_image(images, sess, epoch):
    generated = sess.run(images)

    with open('gen_image\\3th_test\\' + str(epoch) + '.jpeg', 'wb') as f:
        f.write(generated)

config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.7)
)

with tf.Session(config=config) as sess:
    # 시작 시간 체크
    stime = time.time()

    m = DCGAN(sess, batch_size=batch_size)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    best_loss_val = np.infty
    check_since_last_progress = 0  # early stopping 조건을 만족하지 않은 횟수
    max_checks_without_progress = 100  # 특정 횟수 만큼 조건이 만족하지 않은 경우
    best_model_params = None  # 가장 좋은 모델의 parameter 값을 저장하는 변수
    epoch = 1

    print('Data Loading ...')
    total_x = read_data()
    print('Data Loaded!!!')

    print('Learning Started!')

    while True:
        epoch_stime = time.time()
        g_tot_loss, d_tot_loss = 0., 0.

        '''train part'''
        for start_idx in range(0, 800, batch_size):
            g_loss, d_loss, *_ = m.train(total_x[start_idx: start_idx+batch_size], np.random.randn(batch_size, 100))
            g_tot_loss += g_loss / batch_size
            d_tot_loss += d_loss / batch_size

        # monitoring factors
        mon_epoch_list.append(epoch)
        mon_value_list[0].append(g_tot_loss)
        mon_value_list[1].append(d_tot_loss)

        epoch_etime = time.time()
        print('epoch :', epoch, ', g_loss :', round(g_tot_loss, 8), ', d_loss :', round(d_tot_loss, 8), ', time :',
              round(epoch_etime - epoch_stime, 6))
        drawnow(monitor_train_cost)

        if epoch % 100 == 0:
            create_image(m.generate(np.random.randn(batch_size, 100)), epoch)
            # save_image(m.sample_images(), sess, epoch)

        '''early stopping condition check'''
        if d_tot_loss < best_loss_val:
            best_loss_val = d_tot_loss
            check_since_last_progress = 0
            best_model_params = get_model_params()
            saver.save(sess, 'gogh_train_log/dcgan_v1.ckpt')
        else:
            check_since_last_progress += 1

        # if check_since_last_progress > max_checks_without_progress:
        #     print('Early stopping!')
        #     break

        epoch += 1

    print('Learning Finished!')

    # 종료 시간 체크
    etime = time.time()
    print('consumption time : ', round(etime-stime, 6))

    print('\nGenerating Started!')

    if best_model_params:
        restore_model_params(best_model_params)

    create_image(m.generate(np.random.randn(batch_size, 100)), 0)
    # save_image(m.sample_images(), sess, 0)

    print('Generating Finished!')