import os
import json
from glob import glob
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.utils.data as data

from util import get_class_names, make_test_augmenters
from dataset import AudioDataset
from models import ModelWrapper
from config import Config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on {device}')

def create_test_loader(conf, input_dir, class_names):
    test_audio_aug, test_image_aug = make_test_augmenters(conf)

    data_dir = 'test_soundscapes'
    test_df = pd.DataFrame()
    data_files = sorted(glob(f'{input_dir}/{data_dir}/*.ogg'))
    assert len(data_files) > 0, f'No files inside {input_dir}/{data_dir}'
    data_files = [os.path.basename(filename) for filename in data_files]
    test_df['filename'] = data_files
    test_df['primary_label'] = class_names[0]
    test_dataset = AudioDataset(
        test_df, conf, input_dir, data_dir,
        class_names, test_audio_aug, test_image_aug, is_test=True)
    print(f'{len(test_dataset)} examples in test set')
    loader = data.DataLoader(
        test_dataset, batch_size=conf.batch_size, shuffle=False,
        num_workers=mp.cpu_count(), pin_memory=False)
    return loader, test_df

def create_model(model_dir, num_classes):
    checkpoint = torch.load(f'{model_dir}/model.pth', map_location=device)
    conf = Config(checkpoint['conf'])
    conf.pretrained = False
    model = ModelWrapper(conf, num_classes)
    model = model.to(device)
    model.load_state_dict(checkpoint['model'])
    return model, conf

def predict_batch(outputs, threshold):
    sigmoid = nn.Sigmoid()
    # multi-label classification:
    # set prediction to 1 if probability >= threshold
    return sigmoid(outputs).cpu().numpy() >= threshold

def test(loader, model, num_classes, threshold):
    preds = np.zeros((len(loader.dataset), num_classes), dtype=np.float32)
    start_idx = 0
    model.eval()
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            end_idx = start_idx + outputs.shape[0]
            preds[start_idx:end_idx] = predict_batch(outputs, threshold)
            start_idx = end_idx
    return preds

def save_results(input_dir, df, preds, class_names):
    class_map = {name: i for i, name in enumerate(class_names)}
    with open(f'{input_dir}/scored_birds.json') as json_file:
        birds = json.load(json_file)

    pred_idx = 0
    row_ids = []
    targets = []
    for filename in df['filename'].values:
        for clip_idx in range(12):
            end_time = 5*(clip_idx + 1)
            clip_preds = preds[pred_idx]
            pred_idx += 1
            for bird in birds:
                key = f'{filename[:-4]}_{bird}_{end_time}'
                pred = clip_preds[class_map[bird]] > 0
                row_ids.append(key)
                targets.append(pred)
    subm = pd.DataFrame()
    subm['row_id'] = row_ids
    subm['target'] = targets
    subm.to_csv('submission.csv', index=False)
    print('Saved submission.csv')

def run(input_dir, model_dir, threshold=0.3):
    meta_file = os.path.join(input_dir, 'train_metadata.csv')
    train_df = pd.read_csv(meta_file, dtype=str)
    class_names = np.array(get_class_names(train_df))
    num_classes = len(class_names)

    model, conf = create_model(model_dir, num_classes)
    loader, df = create_test_loader(conf, input_dir, class_names)
    assert len(loader.dataset) == 12*df.shape[0]
    preds = test(loader, model, num_classes, threshold)
    save_results(input_dir, df, preds, class_names)

if __name__ == '__main__':
    run('../input', './')

