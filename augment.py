import random
import audiomentations as AA
from audiomentations.core.transforms_interface import BaseWaveformTransform
import noisereduce as nr
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class DenoiseTransform(BaseWaveformTransform):

    def __init__(self, num_fft, p=0.5):
        super().__init__(p)
        self.num_fft = num_fft

    def apply(self, samples, sr):
        level = random.random()
        return nr.reduce_noise(y=samples, sr=sr, prop_decrease=level, n_fft=self.num_fft)

def make_audio_augmenter(conf):
    p = conf.audio_aug_prob
    if p <= 0:
        return None
    aug_list = []
    aug_list = [
        AA.OneOf([
            AA.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.2*p),
            AA.AddGaussianSNR(p=0.2*p),
            AA.ClippingDistortion(min_percentile_threshold=20, max_percentile_threshold=40, p=0.2*p),
            DenoiseTransform(conf.num_fft, p=0.5*p)
        ], p=p),
        AA.Gain(min_gain_in_db=-12, max_gain_in_db=12, p=0.2*p),
    ]

    aug_list.extend([
        AA.OneOf([
            AA.TimeStretch(min_rate=0.8, max_rate=0.9, leave_length_unchanged=True, p=0.2*p),
            AA.PitchShift(min_semitones=-2, max_semitones=-1, p=0.2*p),
            AA.PolarityInversion(p=0.2*p),
        ], p=p),
        AA.Shift(min_fraction=0.5, max_fraction=0.5, p=0.2*p),
    ])
    return AA.Compose(aug_list)

def make_image_augmenter(conf):
    p = conf.image_aug_prob
    if p <= 0:
        return A.Compose([
            A.Normalize(),
            ToTensorV2()
        ])

    aug_list = []
    if conf.max_cutout > 0:
        aug_list.extend([
            A.CoarseDropout(
                max_holes=conf.max_cutout, min_holes=1,
                max_height=conf.num_mels//10, max_width=conf.spectrogram_width//10,
                min_height=4, min_width=4, p=0.2*p),
        ])

    aug_list.extend([
        A.OneOf([
            A.MotionBlur(p=0.2*p),
            A.MedianBlur(blur_limit=3, p=0.1*p),
            A.Blur(blur_limit=3, p=0.1*p),
        ], p=0.2*p),
        A.Perspective(p=0.2*p),
    ])

    aug_list.extend([
        A.GaussNoise(p=0.2*p),
        A.OneOf([
            A.Sharpen(p=0.2*p),
            A.Emboss(p=0.2*p),
            A.RandomBrightnessContrast(p=0.2*p),
        ], p=0.3*p),
    ])

    aug_list.extend([
        A.Normalize(),
        ToTensorV2()
    ])

    return A.Compose(aug_list)

def make_train_augmenters(conf):
    return make_audio_augmenter(conf), make_image_augmenter(conf)
