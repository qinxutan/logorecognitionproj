# **Siamese Network for Logo Recognition**

## **Overview**
This repository contains code for a **Siamese neural network** designed for logo recognition. The network is trained and evaluated on datasets aimed at distinguishing legitimate logos from phishing indicators.

## **Features**
- **Siamese Neural Network**: Utilizes a Siamese architecture to learn embeddings for logos.
- **PhishIntention Dataset**: Initial dataset consisting of 2173 classes with a minimum/average/maximum of 3 images per class, used for pretraining.
- **SPF Dataset**: Supplementary dataset of 2173 classes with a minimum/average/maximum of 1 image per class, used for evaluation.
- **Training and Evaluation**: Includes scripts and configurations for training the model on combined datasets and evaluating its performance.

## **Requirements**
- **Python 3.9**
- **TensorFlow** 
- **PyTorch**
- **OpenCV**

## **Usage**
### **Training**
1. **Data Preparation**: Upload datasets into the input folder.
2. **Configuration**: Adjust parameters in `config.py` as necessary for training on combined datasets.
3. **Training Script**: Execute the training script (`train.py`) to train the Siamese network on the combined dataset.

    ```bash
    python train.py
    ```
### **Evaluation**
1. **Inference**: Perform inference on the datasets using the trained model to evaluate performance.
2. **Evaluation Metrics**: Evaluate using predefined thresholds and metrics specified in `config.py`.

    ```bash
    python inference.py
    ```

## **Contributors**
- **Maintainer**: [Tan Qin Xu](https://github.com/qinxutan)

## **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
