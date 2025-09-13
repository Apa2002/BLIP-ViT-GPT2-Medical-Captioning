# Medical Image Captioning with BLIP (ROCOv2 Dataset)

This project implements **medical image captioning** using the **BLIP (Salesforce BLIP-Base)** model on the **ROCOv2 dataset**.  
The goal is to generate textual descriptions for medical images, including X-rays and CT scans.

---

## Project Steps

### 1. Data Loading and Preprocessing
- Load medical images and captions from **ROCOv2**.
- Resize images to **224×224 pixels**.
- Process captions using **BlipProcessor**.
- Split dataset into **training** and **test** sets.
- Use **10% of training data** for faster training on limited GPU.

---

### 2. Model Architecture (BLIP + LoRA)
- **Encoder**: Vision Transformer (BLIP)  
- **Decoder**: Text generation with GPT  
- **LoRA Fine-Tuning**: Applied to attention layers (query & key) to reduce trainable parameters and memory usage.

---

### 3. Training
- Optimizer: **AdamW**  
- Scheduler: **Linear**  
- Batch size: **3**  
- Epochs: **4 and 6** (10% of data)  
- Save model and optimizer states after each epoch.

---

### 4. Evaluation
- Generate captions for **test images**.  
- Metrics:
  - **BLEU (1-4)**  
  - **METEOR**  
  - **ROUGE-L**  
  - **CIDEr**

---

### 5. Results
| Metric      | BLIP 4 Epochs | BLIP 6 Epochs | Without Fine-tune |
|------------|---------------|---------------|-----------------|
| BLEU-1     | 0.0470        | 0.0545        | 0.0433          |
| BLEU-2     | 0.0222        | 0.0255        | 0.0151          |
| METEOR     | 0.0270        | 0.0278        | 0.0171          |
| ROUGE-L    | 0.0882        | 0.0892        | 0.0812          |
| CIDEr      | 0.0387        | 0.0401        | 0.0056          |

---

## Future Improvements
- Train with **larger data** (>10%) for better results.  
- Explore other **Vision Transformer encoders**.  
- Tune **hyperparameters** and use **longer epochs**.  
- Apply **data augmentation**.

---

## Dataset
**ROCOv2: Radiology Objects in COntext Version 2**  
- 79,789 radiological images with captions and clinical concepts  
- Seven clinical modalities, manually curated medical concepts  
- Suitable for **image captioning** and **multi-label classification**

[Dataset Link](https://zenodo.org/records/10821435)  

**Citation (APA):**  
> Johannes Rückert, Louise Bloch, Raphael Brüngel, Ahmad Idrissi-Yaghir, Henning Schäfer, Cynthia S. Schmidt, Sven Koitka, Obioma Pelka, Asma Ben Abacha, Alba Garcia Seco de Herrera, Henning Müller, Peter A. Horn, Felix Nensa, & Christoph M. Friedrich. (2023). ROCOv2: Radiology Objects in COntext Version 2, An Updated Multimodal Image Dataset [Data set]. *Scientific Data (2.0.1)*. Zenodo. https://doi.org/10.5281/zenodo.10821435

---

## Requirements
```bash
torch>=2.0
transformers>=4.10.3
datasets>=2.20.0
peft
bitsandbytes
pycocoevalcap
evaluate
sacrebleu
nltk
rouge-score
numpy
pandas
matplotlib
tqdm
Pillow
