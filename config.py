import torch

max_duration = 7.0
sampling_rate = 44100
debug = True
train_path = "train.csv"
valid_path = "val.csv"
batch_size = 8
num_workers = 4
lr = 1e-3
weight_decay = 1e-3
patience = 2
factor = 0.5
epochs = 5
seq_max_len=200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

caption_model_name = "gpt2"
image_embedding = 2048
caption_encoder_trainable = False
audio_encoder_trainable = False

pretrained_clap_path = (
    "checkpoint/msclap/CLAP_weights_2023.pth"  # for both image encoder and text encoder
)
temperature = 1.0

# image size
size = 224

# for projection head; used for both image and text encoders
num_projection_layers = 1
projection_dim = 256
dropout = 0.1
