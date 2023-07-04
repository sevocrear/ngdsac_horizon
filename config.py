from dds_classes import orientation_dds, acceleration_dds, velocity_dds, position_dds
from ImageDDS._sample import Image
from cyclonedds.qos import Qos, Policy
from cyclonedds.util import duration

import numpy as np


# Имена топиков для логирования
topic_list = [
    'camera_images',
]

topic_classes = [
    Image,
]

# имена файлов для записи
# без указания стандартного будет название топика + '_log'
logger_ns = {
    "camera_images": 'camera_images_clr_log_2023-06-29-18-20-18',
}

is_video = {
    'camera_images': False,
}

video_ns = {
    'camera_images': 'horizon_2023-06-29-18-20-18/camera_images_clr_log_2023-06-29-18-20-18.mp4',
}

# Qos-ы для CycloneDDS
qos = {
    'camera_images': Qos(Policy.Reliability.Reliable(max_blocking_time=duration(seconds=1))),
}

# время между сеансами чтения
break_time = 1 / 30

# Функция для конвертации image типа в np.array
def dds_to_video(dds_sample):
    return np.frombuffer(dds_sample.flatten_image, dtype=np.uint8).reshape((1080, 1280, 3))
