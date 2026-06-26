# Cat and Dog Image Classification using CNN

A deep learning project that classifies images as **Cat** or **Dog** using a **Convolutional Neural Network (CNN)** built with **TensorFlow/Keras**. The trained model is deployed as an interactive **Streamlit** web application, allowing users to upload an image and receive real-time predictions.

---

## Project Overview

Image classification is one of the most common applications of Deep Learning in Computer Vision. This project develops a CNN model capable of identifying whether an uploaded image belongs to the **Cat** or **Dog** class. The workflow includes data preprocessing, model training, evaluation, prediction on unseen images, and deployment through Streamlit.

---

## Live Application

Launch Cat and Dog Classifier

https://cat-dog-image-classification-cnn-xq4on9e5cfbuwmreixkglh.streamlit.app/

---

## Objectives

- Understand image classification using CNN.
- Preprocess and augment image data.
- Build and train a CNN model.
- Evaluate model performance.
- Predict unseen images.
- Deploy the trained model using Streamlit.

---

## Project Workflow

### Phase 1: Model Development

- Import required libraries
- Load image dataset
- Image preprocessing
- Data augmentation
- Create training and testing datasets
- Build CNN architecture
- Compile the model
- Train the model
- Visualize training accuracy and loss

### Phase 2: Model Evaluation

- Evaluate the trained model
- Generate Confusion Matrix
- Test with unseen images
- Save the trained model
- Load the saved model

### Phase 3: Streamlit Deployment

- Develop an interactive Streamlit web application
- Upload images for prediction
- Display predicted class
- Show prediction confidence score

---

## Model Evaluation

The trained CNN model was evaluated using:

- Accuracy
- Confusion Matrix
- Unseen Image Predictions

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Pillow
- Streamlit

---

## Project Structure

```text
cat-dog-image-classification-cnn/
│
├── Cat&Dog CNN Model.ipynb
├── catdog_app.py
├── cat_dog_cnn.h5
├── requirements.txt
├── README.md

```

---

## Run the Project

### Clone the repository

```bash
git clone https://github.com/shifanacc/cat-dog-image-classification-cnn.git
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Launch the Streamlit application

```bash
streamlit run catdog_app.py
```

---

## Streamlit Application

The Streamlit application enables users to:

- Upload a cat or dog image
- Perform automatic image preprocessing
- Predict the image class
- Display the confidence score in real time

---

## Future Improvements

- Improve prediction accuracy using Transfer Learning (MobileNetV2, EfficientNet, or ResNet50)
- Add Grad-CAM visualization for model interpretability
- Support multiple image uploads

---
