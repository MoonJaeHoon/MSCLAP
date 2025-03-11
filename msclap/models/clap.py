import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel
from .audio import get_audio_encoder
from typing import List


# For CLIP, projection(linear) -> gelu -> fc(linear) -> dropout -> skip-connection (x + projected) -> layer_norm
class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float=0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds

class AudioEncoder(nn.Module):
    def __init__(self, audioenc_name:str, d_in: int, d_out: int, sample_rate: int, window_size: int,
            hop_size: int, mel_bins: int, fmin: int, fmax: int, classes_num: int, trainable: bool = False,) -> None:
        super().__init__()

        audio_encoder = get_audio_encoder(audioenc_name)

        self.base = audio_encoder(
            sample_rate, window_size,
            hop_size, mel_bins, fmin, fmax,
            classes_num, d_in)
        # for p in self.base.parameters():
        #     p.requires_grad = trainable

        self.projection = Projection(d_in, d_out)

    def forward(self, x):
        out_dict = self.base(x)
        audio_features, audio_classification_output = out_dict['embedding'], out_dict['clipwise_output']
        projected_vec = self.projection(audio_features)
        return projected_vec, audio_classification_output

class TextEncoder(nn.Module):
    def __init__(self, d_out: int, text_model: str, transformer_embed_dim: int, trainable: bool = False) -> None:
        super().__init__()
        self.text_model = text_model
        self.base = AutoModel.from_pretrained(text_model)

        if 'clip' in text_model:
            self.clip_text_projection = self.base.text_projection
            self.base = self.base.text_model
            if 'base' in text_model:
                transformer_embed_dim = 512
        
        # for p in self.base.parameters():
        #     p.requires_grad = trainable

        
        self.projection = Projection(transformer_embed_dim, d_out)

    def forward(self, x):
        if 'clip' in self.text_model:
            pooled_output = self.base(**x)[1] # get pooled output
            out = self.clip_text_projection(pooled_output)  # get CLS token output
        elif 'gpt' in self.text_model:
            batch_size = x['input_ids'].shape[0]
            hidden_states = self.base(**x)[0] # (batch_size=4, seq_len, 768)

            sequence_lengths = torch.ne(x['input_ids'], 0).sum(-1) - 1 # tensor([13, 14, 18, 17])
            out = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths] # [batch_size, 768] = [4, 768]
        else:
            out = self.base(**x)[0]
            out = out[:, 0, :]  # get CLS token output
        
        projected_vec = self.projection(out)

        return projected_vec

class CLAP(nn.Module):
    def __init__(self,
                # audio
                audioenc_name: str,
                sample_rate: int, 
                window_size: int, 
                hop_size: int, 
                mel_bins: int, 
                fmin: int, 
                fmax: int, 
                classes_num: int, 
                out_emb: int,
                # text
                text_model: str,
                transformer_embed_dim: int,
                # common
                d_proj: int,
                ):
        super().__init__()

        
        self.audio_encoder = AudioEncoder(
            audioenc_name, out_emb, d_proj,
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num)

        self.caption_encoder = TextEncoder(
            d_proj, text_model, transformer_embed_dim
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) # 2.659

    def forward(self, audios:torch.Tensor, texts:List):
        audio_embed, _ = self.audio_encoder(audios)
        caption_embed = self.caption_encoder(texts)


        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(audio_embed)
        text_embeddings = self.text_projection(caption_embed)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()

        return caption_embed, audio_embed, self.logit_scale.exp()

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


if __name__ == '__main__':
    audios = torch.randn(4,600) # waveform (batch_size, audio_len)
    texts = torch.randint(5, 300, size=(4, 25))
    attention_mask = torch.ones(8, 25)
    batch = {
        'audio': audios,
        'texts': texts,
    }
    
    # images = torch.randn(8, 3, 224, 224)
    # input_ids = torch.randint(5, 300, size=(8, 25))
    # attention_mask = torch.ones(8, 25)
    # batch = {
    #     'image': images,
    #     'input_ids': input_ids,
        # 'attention_mask': attention_mask
    # }

    clap = CLAP()
    audio_embeds, _ = clap.audio_encoder(audios)
    text_embeds = clap.text_encoder(texts)
    
    loss = clap(batch)
    print("")