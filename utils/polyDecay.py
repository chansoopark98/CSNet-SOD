import numpy as np

def poly_decay(lr=3e-4, max_epochs=100, warmup=False):
    """
    poly decay.
    :param lr: 초기 learning rate
    :param max_epochs: 에폭횟수
    :param warmup: warm up 사용 여부 ( 초기에 lr을 높게 가져가는 방법 )
    :return: 현재 lr
    """
    max_epochs = max_epochs - 5 if warmup else max_epochs

    def decay(epoch):
        lrate = lr * (1 - np.power(epoch / max_epochs, 0.9))
        return lrate

    return decay