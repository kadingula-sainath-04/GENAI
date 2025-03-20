# Text-to-Image Generation with Stable Diffusion using hugging face auth token

## Overview
This project generates images from text prompts using Stable Diffusion. It also includes a translation feature powered by the Google Translate API.

## Features
- Text-to-Image generation using Stable Diffusion
- Text translation via Google Translate API
- Runs on Google Colab with GPU support

## Installation
Run the following commands to set up the environment:

```sh
!pip install googletrans==4.0.0-rc1
!pip install --upgrade diffusers transformers torch gradio -q
```

## Usage
### **1️⃣ Load Hugging Face Model**
```python
from huggingface_hub import login
login("your_huggingface_token")
```

### **2️⃣ Initialize Model**
```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2")
pipe.to("cuda")
```

### **3️⃣ Generate an Image**
```python
def generate_image(prompt, model):
    image = model(prompt, num_inference_steps=35, guidance_scale=9).images[0]
    image = image.resize((900, 900))
    return image
```

### **4️⃣ Translate Text**
```python
from googletrans import Translator

def get_translation(text, dest_lang="en"):
    translator = Translator()
    translated_text = translator.translate(text, dest=dest_lang)
    return translated_text.text
```

## Running in Google Colab
1. Open a new Google Colab notebook
2. Install dependencies using the commands above
3. Authenticate with Hugging Face
4. Load and test the Stable Diffusion model
5. Generate images and translate text as needed

## Dependencies
- Python 3.8+
- Hugging Face Transformers
- Diffusers
- Google Translate API
- Torch (CUDA-enabled for GPU acceleration)
