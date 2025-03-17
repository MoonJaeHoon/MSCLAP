# Implementation for Training Code for MSCLAP

## Source
**Paper**
- https://arxiv.org/pdf/2309.05767

**Github**
- https://github.com/microsoft/CLAP

## Setup
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