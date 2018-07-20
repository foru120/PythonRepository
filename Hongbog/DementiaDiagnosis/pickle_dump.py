import numpy as np
import pickle, os, ntpath, cv2

tot_list = []
img_list = []
label_list = []
pickle_dict = {}
load_path = ''

def _load_data(path):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            imgpath = os.path.join(root, file)
            tot_list.append(imgpath)

    np.random.shuffle(tot_list)

    for img in tot_list:
        image = cv2.imread(img)
        image = cv2.resize(image, (192, 192), cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = [1, 0] if ntpath.basename(img)[0] == '1' else [0, 1]

        img_list.append(image)
        label_list.append(label)
        count += 1

    print('>> 총 {} 데이터 loading !'.format(count))

    tot_img = np.asarray(img_list)

    pickle_dict[b'labels'] = label_list
    pickle_dict[b'data'] = tot_img


    with open('database/rgb_data/gelontoxon_data', 'wb') as data:
        pickle.dump(pickle_dict, data)

    print('>> pickle dumpling 완료 ! ')


if __name__ == '__main__':
    path = '/home/kyh/dataset/gelontoxon/'
    _load_data(path)