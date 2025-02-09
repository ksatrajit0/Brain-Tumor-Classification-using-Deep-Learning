import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from src.training import load_data

if __name__ == "__main__":
    data_dir = "/kaggle/working/cleaned"
    df = load_data(data_dir)

    model_path = 'models/effnetv2b0_multi_head_attention_dataset1.h5'
    model = load_model(model_path)

    gen = ImageDataGenerator()
    test_gen = gen.flow_from_dataframe(test_df, x_col='Picture Path', y_col='Label', target_size=(224, 224), 
                                       class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=16)

    preds = model.predict(test_gen)
    y_pred = np.argmax(preds, axis=1)

    cm = confusion_matrix(test_gen.classes, y_pred)
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, cmap=plt.cm.Reds, fmt='g')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    print(classification_report(test_gen.classes, y_pred, target_names=test_gen.class_indices.keys()))