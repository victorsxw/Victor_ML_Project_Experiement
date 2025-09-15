# Retinal Blood Vessel Segmentation Using U-Net and W-Net

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the implementation and comparison of two deep learning architectures, **U-Net** and **W-Net**, for the automated segmentation of retinal blood vessels. Accurate segmentation is vital for the early detection of eye diseases such as **diabetic retinopathy**, **age-related macular degeneration**, and **glaucoma**.

Our project addresses key challenges in retinal image analysis, including class imbalance and computational complexity, using a diverse dataset of 800 high-resolution fundus images.

## 🎯 Key Results

- **W-Net Superiority**: The W-Net model demonstrated superior performance, achieving a Jaccard Index (IoU) of **0.771** compared to U-Net's 0.764
- **Computational Efficiency**: W-Net required less than half the training time and GPU memory of U-Net
- **Data Leakage Correction**: Identified and corrected a critical data leakage issue, improving IoU score from 66% to 77%

## 📊 Performance Comparison

### Segmentation Quality
| Model | Jaccard Index (IoU) | F1-score | Recall | Precision | Accuracy |
|:---|:---:|:---:|:---:|:---:|:---:|
| **U-Net** | 0.764 | 0.858 | 0.811 | **0.926** | 0.983 |
| **W-Net** | **0.771** | **0.863** | **0.827** | 0.913 | **0.984** |

### Computational Efficiency
| Model | System RAM | GPU RAM | Training Time |
|:---|:---:|:---:|:---:|
| **U-Net** | 5.5 GB | 9.4 GB | 46.6 min |
| **W-Net** | **3.6 GB** | **5.9 GB** | **18.3 min** |

## 🔬 Methodology

### 📁 Dataset
We used the **FIVES dataset**, which includes 800 high-resolution fundus images (2048 × 2048 pixels) categorized into four groups:
- Normal eyes
- Macular degeneration
- Diabetic retinopathy  
- Glaucoma

This dataset provides pixel-level ground-truth annotations for blood vessels, addressing the limitations of smaller datasets.

### 🏗️ Model Architectures

#### U-Net
- Standard convolutional neural network (CNN) for biomedical image segmentation
- Encoder-decoder structure with skip connections
- Retains fine-grained details through skip connections

#### W-Net
- Dual U-Net architecture designed to refine segmentation results
- Requires fewer parameters and computational resources
- More efficient than traditional U-Net variants

### ⚙️ Preprocessing and Training
- **Image Resizing**: All images resized to 384 × 384 pixels
- **Normalization**: Standard image normalization applied
- **Data Split**: Training (60%), validation (20%), test (20%)
- **Loss Function**: Dice Loss to handle class imbalance (vessels occupy small portion of image)

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- CUDA (for GPU acceleration)

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/retinal-vessel-segmentation.git
cd retinal-vessel-segmentation

# Install dependencies
pip install -r requirements.txt
```

### Usage
The code can be run in a Google Colab environment or locally:

1. **Google Colab**: Open the provided notebook link
2. **Local Environment**: Run the Jupyter notebook locally

### 📁 Repository Structure
```
├── Retina Blood Vessel Segmentation - Final.ipynb  # Main implementation notebook
├── results/                                        # Model predictions and evaluation results
│   ├── unet_predictions/                          # U-Net segmentation results
│   ├── wnet predictions/                          # W-Net segmentation results
│   ├── unet_evaluation_results.json              # U-Net metrics
│   └── wnet_evaluation_results.json              # W-Net metrics
└── README.md                                      # This file
```

## 🎯 Challenges and Future Work

### Current Challenges
- Limited computational resources during development
- Lower segmentation performance for glaucoma patients (likely due to image blurring)
- Learning curve with complex deep learning architectures

### Future Directions
- **Glaucoma-specific preprocessing**: Improve segmentation for glaucoma patients through targeted preprocessing techniques
- **Multi-task learning**: Build a model that can both segment blood vessels and classify eye diseases
- **Generalizability testing**: Evaluate model performance on other publicly available fundus image datasets
- **Real-time deployment**: Optimize models for real-time clinical applications

## 📚 Resources and Links

### 🔗 External Resources
- **[Google Colab Notebook](https://colab.research.google.com/drive/1miwTCayM-WfZgFsJR71ORNs2Sef6pjtD?usp=drive_link)**: Interactive notebook with complete implementation
- **[Project Google Drive](https://drive.google.com/drive/folders/1IkVi8Gc92oVy3x943A8ANCTV3Ks-j0kI?usp=sharing)**: Project files and documentation
- **[Results Google Drive](https://drive.google.com/drive/folders/1MRcseWLJ_yZhtunCyk1zLq5gA32ebB2w?usp=sharing)**: Model predictions and evaluation results

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Contact

For questions or collaborations, please open an issue or contact the project maintainers.

---

*This project was developed as part of a machine learning course at York University (EECS 6240).*