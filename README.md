# Medical Image Captioning with BLIP and ViT-GPT2 (ROCOv2 Dataset)

This project implements **image captioning models** on medical images using the **ROCOv2 dataset**.  
Two main architectures are explored:
1. **BLIP (Salesforce BLIP-Base) with LoRA fine-tuning**
2. **ViT-GPT2 (ydshieh/vit-gpt2-coco-en) with LoRA fine-tuning**

The models generate textual descriptions (captions) for medical images such as X-rays and CT scans.

---

## Project Steps

### 1. Data Loading and Preprocessing
- Medical images and their corresponding captions are loaded from **ROCOv2**.
- Images are resized to **224×224 pixels**.
- Captions are tokenized using **GPT-2 tokenizer** (for ViT-GPT2) or processed with **BlipProcessor** (for BLIP).
- Dataset is split into **training and test sets**.
- Only **10% of training data** is used to reduce training time on limited GPU resources.

---

### 2. Model Architectures
#### BLIP (Base) + LoRA
- Encoder: Vision Transformer (BLIP)
- Decoder: Text generation with GPT
- LoRA fine-tunes only specific layers → reduces trainable parameters and memory usage.

#### ViT-GPT2 + LoRA
- Encoder: Vision Transformer (ViT)
- Decoder: GPT-2
- LoRA applied to decoder only to optimize efficiency.

---

### 3. Training
- Optimizer: **AdamW**
- Scheduler: **Linear**
- Epochs: **4 and 6** for 10% of data
- Training includes **checkpoint saving** after each epoch.

---

### 4. Evaluation
- Captions are generated for **test images**.
- Performance metrics:
  - **BLEU (1-4)**
  - **METEOR**
  - **ROUGE-L**
  - **CIDEr**
- Comparison between **generated captions** and **ground truth captions**.

---

### 5. Results
- Increasing epochs improves all metrics slightly (BLEU, METEOR, ROUGE, CIDEr).
- Fine-tuned BLIP outperforms ViT-GPT2 on medical image captions.
- Example outputs:
  - Some captions correctly detect image modality (CT, X-ray), but may miss fine-grained details like pathology.
- Metrics summary for 10% data:

| Metric      | BLIP 4 Epochs | BLIP 6 Epochs | ViT-GPT2 6 Epochs | Without Fine-tune |
|------------|---------------|---------------|------------------|-----------------|
| BLEU-1     | 0.0470        | 0.0545        | 0.0026           | 0.0433          |
| BLEU-2     | 0.0222        | 0.0255        | 0.0015           | 0.0151          |
| METEOR     | 0.0270        | 0.0278        | 0.0063           | 0.0171          |
| ROUGE-L    | 0.0882        | 0.0892        | 0.0195           | 0.0812          |
| CIDEr      | 0.0387        | 0.0401        | 0.0026           | 0.0056          |

---

## Future Improvements
- Train with **larger percentage of data** (>10%) for better results.
- Explore other **Vision Transformer encoders**.
- Experiment with **hyperparameter tuning and longer epochs**.
- Apply **data augmentation** to improve generalization.

---

## Dataset
Dataset: [ROCOv2: Radiology Objects in COntext Version 2](https://zenodo.org/records/10821435)  
- 79,789 radiological images with captions and clinical concepts.
- Includes seven clinical modalities and manually curated medical concepts.
- Suitable for image captioning and multi-label classification tasks.  
- Citation (APA):
  > Johannes Rückert, Louise Bloch, Raphael Brüngel, Ahmad Idrissi-Yaghir, Henning Schäfer, Cynthia S. Schmidt, Sven Koitka, Obioma Pelka, Asma Ben Abacha, Alba Garcia Seco de Herrera, Henning Müller, Peter A. Horn, Felix Nensa, & Christoph M. Friedrich. (2023). ROCOv2: Radiology Objects in COntext Version 2, An Updated Multimodal Image Dataset [Data set]. *Scientific Data (2.0.1)*. Zenodo. https://doi.org/10.5281/zenodo.10821435

---

## Requirements
```bash
torch>=2.0
transformers
datasets
nltk
pycocoevalcap
numpy
pandas
matplotlib
tqdm
