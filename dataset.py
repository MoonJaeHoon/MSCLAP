import torch
import torchaudio
import albumentations as A
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import config as CFG

tokenizer = AutoTokenizer.from_pretrained(CFG.caption_model_name)
if 'gpt' in CFG.caption_model_name:
    tokenizer.add_special_tokens({'pad_token': '!'})


class AudioCaptionDataset(Dataset):
    def __init__(self, csv_path, tokenizer, sampling_rate, max_duration):
        """
        Args:
            dataframe (pd.DataFrame): 'audio_path'와 'caption' 컬럼을 가진 데이터프레임
        """
        self.dataframe = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_duration = max_duration
        self.sr = sampling_rate

    def load_audio(self, filename):
        ext = filename.split(".")[-1]

        audio, in_sr = torchaudio.load(filename, format=ext)

        if in_sr != self.sr:
            resample_tf = T.Resample(in_sr, self.sr)
            audio = resample_tf(audio)

        return audio

    def load_audio_into_tensor(self, audio_path):
        r"""Loads audio file and returns raw audio."""
        # Randomly sample a segment of max_duration from the clip or pad to match duration
        audio_time_series = self.load_audio(audio_path)
        audio_time_series = audio_time_series.reshape(-1)

        # audio_time_series is shorter than predefined audio duration,
        # so audio_time_series is extended
        if self.max_duration * self.sr >= audio_time_series.shape[0]:
            repeat_factor = int(
                np.ceil((self.max_duration * self.sr) / audio_time_series.shape[0])
            )
            # Repeat audio_time_series by repeat_factor to match max_duration
            audio_time_series = audio_time_series.repeat(repeat_factor)
            # remove excess part of audio_time_series
            audio_time_series = audio_time_series[0 : self.max_duration * self.sr]
        else:
            # audio_time_series is longer than predefined audio duration,
            # so audio_time_series is trimmed
            start_index = random.randrange(
                audio_time_series.shape[0] - self.max_duration * self.sr
            )
            audio_time_series = audio_time_series[
                start_index : start_index + self.max_duration * self.sr
            ]
        return torch.FloatTensor(audio_time_series)



    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        audio_path = row["audio_path"]
        raw_caption = row["caption"]
        
        
        loaded_audio = load_audio_into_tensor(audio_path, max_duration)
        return loaded_audio, raw_caption


def encode_captions(caption_lst: List) -> Dict[str, torch.Tensor]:
    encoded_captions = tokenizer(
    caption_lst, padding=True, truncation=True, max_length=150, return_tensors='pt',
    )
    return encoded_captions


def collate_fn(batch):
    """
    배치를 묶는 함수
    Args:
        batch (list of tuples): (audio_path, caption) 리스트

    Returns:
        audios (tensor): 오디오 
        captions (tensor): 캡션 
    """
    
    loaded_audios, raw_captions = zip(*batch) # loaded_audios, encoded_captions
    encoded_captions = encode_captions(raw_captions)

    return loaded_audios, encoded_captions # [torch.stack(loaded_audios, encoded_captions), ...]


def get_data_loader(csv_path, max_duration, mode="train"):
    dataset = AudioCaptionDataset(csv_path, max_duration)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if mode == "train" else False,
        collate_fn=collate_fn,
    )
    return dataloader
