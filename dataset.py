import os
import cv2
import random
import librosa
import numpy as np
import torch.utils.data as data
import warnings
warnings.simplefilter("ignore")


class AudioDataset(data.Dataset):
    def __init__(
            self, df, conf, input_dir, imgs_dir,
            class_names, audio_transform, image_transform, is_val=False, is_test=False):
        self.conf = conf
        self.audio_transform = audio_transform
        self.image_transform = image_transform
        self.is_test = is_test
        self.is_val = is_val

        files = df['filename']
        assert isinstance(files[0], str), (
            f'column {df.columns[0]} must be of type str')
        self.files = [os.path.join(input_dir, imgs_dir, f) for f in files]

        labels = df['primary_label']
        num_samples = len(files)
        self.num_classes = len(class_names)
        class_map = {class_names[i]: i for i in range(self.num_classes)}
        if is_test or is_val:
            self.sample_count =  5*conf.sample_rate
            self.spectrogram_width = conf.spectrogram_width
        else:
            self.sample_count =  conf.duration*conf.sample_rate
            assert conf.duration%5 == 0
            self.spectrogram_width = conf.spectrogram_width*conf.duration//5
        self.labels = np.zeros((num_samples, self.num_classes), dtype=np.float32)
        for i in range(num_samples):
            row_labels = [class_map[token] for token in labels[i].split(' ')]
            self.labels[i, row_labels] = 1.0

    def get_spectrogram(self, sound):
        conf = self.conf
        spec = librosa.feature.melspectrogram(
            y=sound, sr=conf.sample_rate, n_fft=conf.num_fft, hop_length=conf.hop_length,
            n_mels=conf.num_mels,  fmin=conf.min_freq, fmax=conf.max_freq)
        img = librosa.power_to_db(spec, ref=np.max)
        img -= img.min()
        max_val = img.max()
        if max_val != 0:
            img /= max_val
        img *= 255
        img = img.round().astype(np.uint8)
        img = cv2.resize(
            img, (self.spectrogram_width, conf.num_mels),
            interpolation=cv2.INTER_AREA)

        return np.stack((img, img, img), axis=2)

    def load_clip(self, filename, offset, duration):
        assert os.path.isfile(filename)
        sc = self.sample_count
        sound, rate = librosa.load(
            filename, sr=self.conf.sample_rate, offset=offset, duration=duration)
        assert rate == self.conf.sample_rate
        while (sound.shape[0] < sc):
            # pad to required length by duplicating data
            sound = np.hstack((sound, sound[:(sc - sound.shape[0])]))
        return sound

    def load_training_clip(self, index):
        conf = self.conf
        label = self.labels[index]
        filename = self.files[index]
        total_duration = librosa.get_duration(filename=filename)
        if total_duration < conf.duration:
            offset = 0
        else:
            offset = random.uniform(0, total_duration - conf.duration)
        result = self.load_clip(filename, offset, conf.duration)
        # apply label smoothing
        label = np.abs(label - conf.label_smoothing)
        return result, label

    def __getitem__(self, index):
        conf = self.conf
        if self.is_test:
            filename = self.files[index//12]
            label = self.labels[index//12]
            clip_idx = index%12
            offset = clip_idx*5
            sound = self.load_clip(filename, offset, 5)
        elif self.is_val:
            filename = self.files[index]
            label = self.labels[index]
            sound = self.load_clip(filename, 0, 5)
        else:
            sound, label = self.load_training_clip(index)

        if self.audio_transform:
            sound = self.audio_transform(samples=sound, sample_rate=conf.sample_rate)

        img = self.get_spectrogram(sound)
        if self.image_transform:
            img = self.image_transform(image=img)['image']
        return img, label

    def __len__(self):
        if self.is_test:
            # there are 12 five-second clips in each soundscape
            return 12*len(self.files)
        return len(self.files)
