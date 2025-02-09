import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os
import random

# Load your pre-trained model
model_path = '/kaggle/working/effnetv2b0_multi_head_attention_dataset1.h5'
model = tf.keras.models.load_model(model_path, custom_objects={"Adamax": tf.keras.optimizers.Adamax})

# Function to load and preprocess an image
def load_and_preprocess_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.efficientnet_v2.preprocess_input(img)
    return img

# Load paths and labels from the directory
data_dir = "/kaggle/input/brain-tumor-mri-images-44c"
paths = []
labels = []
folds = os.listdir(data_dir)
for fold in folds:
    condition_path = os.path.join(data_dir, fold)
    all_pic = os.listdir(condition_path)
    for each_pic in all_pic:
        each_pic_path = os.path.join(condition_path, each_pic)
        paths.append(each_pic_path)
        labels.append(fold.split(' ')[0])

# Convert paths and labels to pandas Series
import pandas as pd
pseries = pd.Series(paths, name='Picture Path')
lseries = pd.Series(labels, name='Label')

# Filter and sample images for each label
all_imgs = []
for label in lseries.unique():
    label_imgs = pseries[lseries == label].tolist()
    sampled_imgs = random.sample(label_imgs, 10)
    all_imgs.extend(sampled_imgs)

random.shuffle(all_imgs)

# Function to generate Grad-CAM
def generate_gradcam(img_array, model, layer_name):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))

    cam = np.zeros(conv_outputs.shape[1:3], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_outputs[0, :, :, i]

    cam = np.maximum(cam, 0)
    cam = cam / cam.max() if cam.max() != 0 else cam

    return cam

# Function to overlay Grad-CAM on image
def overlay_gradcam(img_path, cam):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    cam = cv2.resize(cam, (224, 224))  # Ensure the cam has the same size as the image
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return img, heatmap, superimposed_img

# Get unique labels
unique_labels = lseries.unique()

# Generate and plot Grad-CAM for each image
layer_name = 'top_conv'  # Adjust the layer name if necessary
for img_path in all_imgs:
    img_array = load_and_preprocess_image(img_path)
    cam = generate_gradcam(img_array, model, layer_name)
    img, heatmap, superimposed_img = overlay_gradcam(img_path, cam)
    
    # Get the true label for the image
    true_label = lseries[pseries == img_path].iloc[0]
    
    # Get the predicted label for the image
    preds = model.predict(img_array)
    predicted_class = np.argmax(preds[0])
    pred_class_name = unique_labels[predicted_class]
    
    if true_label == pred_class_name:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.title(f'Original Image\nTrue Label: {true_label}')
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        plt.subplot(1, 3, 2)
        plt.title(f'Grad-CAM Heatmap\nPredicted Label: {pred_class_name}')
        plt.imshow(heatmap)
        
        plt.subplot(1, 3, 3)
        plt.title(f'Superimposed Image\nPredicted Label: {pred_class_name}')
        plt.imshow(superimposed_img)
        
        plt.show()