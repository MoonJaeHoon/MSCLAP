# Implementation for Training Code for MSCLAP

## Source
**Paper**
- https://arxiv.org/pdf/2309.05767

**Github**
- https://github.com/microsoft/CLAP

## Setup
### poetry
```
curl -sSL https://install.python-poetry.org | python3
/root/.local/bin/poetry lock
/root/.local/bin/poetry install
/root/.local/bin/poetry self add poetry-plugin-shell
/root/.local/bin/poetry shell
```

### pip
First, install python 3.8 or higher (3.11 recommended). Then, install CLAP using either of the following:

```
# Install pypi pacakge
pip install msclap

# Or Install latest (unstable) git source
pip install git+https://github.com/microsoft/CLAP.git
```

## Reference
clip-train
- https://github.com/moein-shariatnia/OpenAI-CLIP/blob/master/CLIP.py

clip-official
- https://github.com/openai/CLIP/tree/main/clip

fp16 (mixed-precision)
- https://github.com/openai/CLIP/blob/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1/clip/model.py#L375


## Example

- Zero-Shot Classification and Retrieval

```
from msclap import CLAP

# Load model (Choose between versions '2022' or '2023')
# The model weight will be downloaded automatically if `model_fp` is not specified
clap_model = CLAP(version = '2023', use_cuda=False)

# Extract text embeddings
text_embeddings = clap_model.get_text_embeddings(class_labels: List[str])

# Extract audio embeddings
audio_embeddings = clap_model.get_audio_embeddings(file_paths: List[str])

# Compute similarity between audio and text embeddings 
similarities = clap_model.compute_similarity(audio_embeddings, text_embeddings)
```

- Audio Captioning

```
from msclap import CLAP

# Load model (Choose version 'clapcap')
clap_model = CLAP(version = 'clapcap', use_cuda=False)

# Generate audio captions
captions = clap_model.generate_caption(file_paths: List[str])
```

## To-Do List

[] dataloader
    [] audio padding
    [] encoding caption (return type : tensor or dict)
[] train.py
[] Logger