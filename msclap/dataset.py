import torch
import albumentations as A
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class AudioCaptionDataset(Dataset):
    def __init__(self, csv_path):
        """
        Args:
            dataframe (pd.DataFrame): 'audio_path'와 'caption' 컬럼을 가진 데이터프레임
        """
        self.dataframe = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        return row["audio_path"], row["caption"]

def collate_fn(batch):
    """
    배치를 묶는 함수
    Args:
        batch (list of tuples): (audio_path, caption) 리스트
    
    Returns:
        audio_paths (list): 오디오 경로 리스트
        captions (list): 캡션 리스트
    """
    audio_paths, captions = zip(*batch)
    return list(audio_paths), list(captions)

def get_data_loader(csv_path, batch_size=4, mode='train'):
    dataset = AudioCaptionDataset(csv_path)
    dataloader = DataLoader(
                            dataset, batch_size=batch_size, 
                            shuffle=True if mode == "train" else False, 
                            collate_fn=collate_fn
                            )
    return dataloader

