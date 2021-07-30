import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import os

AUTO = tf.data.experimental.AUTOTUNE

class GenerateDatasets:
    def __init__(self, mode, image_size, batch_size):
        """
        Args:
            mode: 불러올 데이터셋 종류입니다
            data_dir: 데이터셋 상대 경로
            image_size: 백본에 따른 이미지 해상도 크기
            batch_size: 배치 사이즈 크기
            target: target: 학습할 데이터셋 종류 ( HKU or MSRA10K )
        """
        self.mode = mode
        self.image_size = image_size
        self.batch_size = batch_size

        self.number_train = 0
        self.number_valid = 0

        self.train_data, self.number_train, self.valid_data, self.number_valid = self._load_datasets()

    def _load_datasets(self):
        train_img_path = './datasets/HKU-IS/imgs/'
        train_gt_path = './datasets/HKU-IS/gt/'
        valid_img_path = './datasets/MSRA10K/imgs/'
        valid_gt_path = './datasets/MSRA10K/gt/'

        hku_img_path = os.listdir('./datasets/HKU-IS/imgs/')
        hku_gt_path = os.listdir('./datasets/HKU-IS/gt/')

        msra10k_img_path = os.listdir('./datasets/MSRA10K/imgs/')
        msra10k_gt_path = os.listdir('./datasets/MSRA10K/gt/')

        hku_img_list = []
        hku_gt_list = []
        msra10k_img_list = []
        msra10k_gt_list = []

        for i, j in enumerate(hku_img_path):
            hku_img_list.append(train_img_path + j)
            hku_gt_list.append(train_gt_path + j)

        for i, j in enumerate(msra10k_img_path):
            msra10k_img_list.append(valid_img_path + j)
            msra10k_gt_list.append(valid_gt_path + j.split('.')[0] + '.png')



        # hku_img = tf.data.Dataset.list_files('./datasets/'+'HKU-IS/imgs/' + '*', shuffle=False)
        # hku_gt = tf.data.Dataset.list_files('./datasets/'+ 'HKU-IS/gt/' + '*', shuffle=False)
        hku = tf.data.Dataset.from_tensor_slices((hku_img_list, hku_gt_list))

        # msra10k_img = tf.data.Dataset.list_files('datasets/' + 'MSRA10K/imgs/' + '*', shuffle=False)
        # msra10k_gt = tf.data.Dataset.list_files('datasets/' + 'MSRA10K/gt/' + '*', shuffle=False)
        msra10k = tf.data.Dataset.from_tensor_slices((msra10k_img_list, msra10k_gt_list))

        if self.mode == 'HKU':
            number_train = hku.reduce(0, lambda x, _: x + 1).numpy()
            print("학습 데이터 개수:", number_train)

            number_valid = msra10k.reduce(0, lambda x, _: x + 1).numpy()
            print("학습 데이터 개수:", number_valid)

            return hku, number_train, msra10k, number_valid
        else:
            number_train = msra10k.reduce(0, lambda x, _: x + 1).numpy()
            print("학습 데이터 개수:", number_train)

            number_valid = hku.reduce(0, lambda x, _: x + 1).numpy()
            print("학습 데이터 개수:", number_valid)

            return msra10k, number_train, hku, number_valid

    @tf.function
    def preprocess(self, img, labels):
        img = tf.io.read_file(img)
        labels = tf.io.read_file(labels)
        try:
            img = tf.image.decode_png(img, channels=3)
        except:
            img = tf.image.decode_jpeg(img, channels=3)
        labels = tf.image.decode_png(labels, channels=3)

        img = tf.image.resize(img, (self.image_size[0], self.image_size[1]))
        labels = tf.image.resize(labels, (self.image_size[0], self.image_size[1]))

        img = tf.cast(img, dtype=tf.float32)

        labels = labels[:, :, :1]

        labels = tf.where(labels>0, 1, 0)
        labels = tf.cast(labels, dtype=tf.int64)

        img = preprocess_input(img, mode='tf')

        return (img, labels)


    @tf.function
    def augmentation(self, img, labels):
        if tf.random.uniform([]) > 0.5:
            img = tf.image.random_brightness(img, max_delta=0.4)
        if tf.random.uniform([]) > 0.5:
            img = tf.image.random_contrast(img, lower=0.7, upper=1.4)
        if tf.random.uniform([]) > 0.5:
            img = tf.image.random_hue(img, max_delta=0.4)
        if tf.random.uniform([]) > 0.5:
            img = tf.image.random_saturation(img, lower=0.7, upper=1.4)
        if tf.random.uniform([]) > 0.5:
            img = tf.image.flip_left_right(img)
            labels = tf.image.flip_left_right(labels)

        return (img, labels)

    def get_trainData(self, train_data):

        train_data = train_data.map(self.preprocess)
        train_data = train_data.shuffle(100).repeat()
        # self.train_data = self.train_data.map(self.augmentation, num_parallel_calls=AUTO)
        train_data = train_data.map(self.augmentation)
        train_data = train_data.batch(self.batch_size).prefetch(AUTO)

        return train_data

    def get_validData(self, valid_data):

        valid_data = valid_data.map(self.preprocess)
        valid_data = valid_data.repeat()
        valid_data = valid_data.batch(self.batch_size).prefetch(AUTO)
        return valid_data

    def get_testData(self, valid_data):

        valid_data = valid_data.map(self.preprocess)
        valid_data = valid_data.batch(self.batch_size).prefetch(AUTO)
        return valid_data
