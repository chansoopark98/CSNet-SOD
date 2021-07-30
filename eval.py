import tensorflow as tf
from models.model_builder import seg_model_build
from utils.data_generator import GenerateDatasets
from tqdm import tqdm
import argparse
import time
import os
tf.keras.backend.clear_session()

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size",     type=int,   help="배치 사이즈값 설정", default=16)
parser.add_argument("--epoch",          type=int,   help="에폭 설정", default=200)
parser.add_argument("--lr",             type=float, help="Learning rate 설정", default=0.001)
parser.add_argument("--weight_decay",   type=float, help="Weight Decay 설정", default=0.0005)
parser.add_argument("--model_name",     type=str,   help="저장될 모델 이름",
                    default=str(time.strftime('%m%d', time.localtime(time.time()))))
parser.add_argument("--dataset_dir",    type=str,   help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')
parser.add_argument("--checkpoint_dir", type=str,   help="모델 저장 디렉토리 설정", default='./checkpoints/')
parser.add_argument("--tensorboard_dir",  type=str,   help="텐서보드 저장 경로", default='tensorboard')
parser.add_argument("--use_weightDecay",  type=bool,  help="weightDecay 사용 유무", default=True)
parser.add_argument("--load_weight",  type=bool,  help="가중치 로드", default=False)
parser.add_argument("--mixed_precision",  type=bool,  help="mixed_precision 사용", default=True)
parser.add_argument("--distribution_mode",  type=bool,  help="분산 학습 모드 설정", default=True)

args = parser.parse_args()
WEIGHT_DECAY = args.weight_decay
BATCH_SIZE = args.batch_size
EPOCHS = args.epoch
base_lr = args.lr
SAVE_MODEL_NAME = args.model_name
DATASET_DIR = args.dataset_dir
CHECKPOINT_DIR = args.checkpoint_dir
TENSORBOARD_DIR = args.tensorboard_dir
IMAGE_SIZE = (224, 224)
USE_WEIGHT_DECAY = args.use_weightDecay
LOAD_WEIGHT = args.load_weight
MIXED_PRECISION = args.mixed_precision
DISTRIBUTION_MODE = args.distribution_mode

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# mirrored_strategy = tf.distribute.MirroredStrategy()
#
# with mirrored_strategy.scope():
# 여기다 하세요 gpu :0
#     print("Number of devices: {}".format(mirrored_strategy.num_replicas_in_sync))
with tf.device('/device:GPU:0'):
    dataset_config = GenerateDatasets(mode='HKU', image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)

    test_data = dataset_config.get_testData(dataset_config.valid_data)

    test_steps = dataset_config.number_valid // BATCH_SIZE

    model = seg_model_build(image_size=IMAGE_SIZE)
    weight_name = '_0730_best_loss'
    model.load_weights(CHECKPOINT_DIR + weight_name + '.h5')

import matplotlib.pyplot as plt
for x, y in tqdm(test_data, total=test_steps):
    pred = model.predict_on_batch(x)#pred = tf.nn.softmax(pred)
    pred = tf.argmax(pred, -1)
    for i in range(len(pred)):

        plt.imshow(pred[i] * 255)
        plt.show()


