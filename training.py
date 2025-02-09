import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.model import create_model

def plot_class_distribution(df, title):
    class_counts = df['Label'].value_counts()
    plt.figure(figsize=(12, 8))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=90)
    
    for i in range(len(class_counts)):
        plt.text(i, class_counts.values[i] + 1, str(class_counts.values[i]), ha='center')
    
    plt.show()

def load_data(data_dir):
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
    
    return pd.DataFrame({'Picture Path': paths, 'Label': labels})

if __name__ == "__main__":
    data_dir = "/kaggle/working/cleaned"
    df = load_data(data_dir)
    
    train_df, ts_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=42, stratify=df['Label'])
    test_df, valid_df = train_test_split(ts_df, test_size=0.5, shuffle=True, random_state=42, stratify=ts_df['Label'])
    
    plot_class_distribution(train_df, 'Number of Samples in Each Class (Train Set)')
    plot_class_distribution(test_df, 'Number of Samples in Each Class (Test Set)')
    plot_class_distribution(valid_df, 'Number of Samples in Each Class (Validation Set)')

    batch_size = 16
    img_size = (224, 224)
    
    gen = ImageDataGenerator()
    train_gen = gen.flow_from_dataframe(train_df, x_col='Picture Path', y_col='Label', target_size=img_size, 
                                        class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)
    valid_gen = gen.flow_from_dataframe(valid_df, x_col='Picture Path', y_col='Label', target_size=img_size, 
                                        class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)
    test_gen = gen.flow_from_dataframe(test_df, x_col='Picture Path', y_col='Label', target_size=img_size, 
                                       class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=batch_size)
    
    model = create_model(input_shape=(224, 224, 3))
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
    hist = model.fit(train_gen, epochs=50, callbacks=[early_stopping], verbose=1, validation_data=valid_gen, shuffle=False)
    
    model.save('models/effnetv2b0_multi_head_attention_dataset1.h5')
    
    plt.figure(figsize=(20, 8))
    plt.style.use('fivethirtyeight')

    plt.subplot(1, 2, 1)
    plt.plot(hist.history['val_loss'], 'g', label='Validation Loss')
    plt.plot(hist.history['loss'], 'r', label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(hist.history['accuracy'], 'r', label='Training Accuracy')
    plt.plot(hist.history['val_accuracy'], 'g', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.tight_layout()
    plt.show()