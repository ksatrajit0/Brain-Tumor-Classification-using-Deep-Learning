import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout,
    BatchNormalization, Input,
    MultiHeadAttention, Reshape
)
from tensorflow.keras.optimizers import Adamax
from sklearn.metrics import classification_report, confusion_matrix
import itertools

print('Modules loaded successfully.')

data_dir = "/kaggle/working/cleaned"
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

pseries = pd.Series(paths, name='Picture Path')
lseries = pd.Series(labels, name='Label')
df = pd.concat([pseries, lseries], axis=1)

batch_size = 16
img_size = (224, 224)
gen = ImageDataGenerator()

val_accuracies = []
best_histories = []
best_val_index = None
best_val_score = 0

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_index, val_index in kf.split(df):
    train_df, val_df = df.iloc[train_index], df.iloc[val_index]

    train_gen = gen.flow_from_dataframe(train_df, x_col='Picture Path', y_col='Label', target_size=img_size, 
                                       class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)

    valid_gen = gen.flow_from_dataframe(val_df, x_col='Picture Path', y_col='Label', target_size=img_size, 
                                       class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)

    def create_model(input_shape, num_classes=15, learning_rate=0.0001):
        inputs = Input(shape=input_shape, name="Input_Layer")
        
        base_model = EfficientNetV2B0(weights='imagenet', input_tensor=inputs, include_top=False)
        
        x = base_model.output
        height, width, channels = x.shape[1], x.shape[2], x.shape[3]
        
        x = Reshape((height * width, channels), name="Reshape_to_Sequence")(x)
        
        attention_output = MultiHeadAttention(num_heads=8, key_dim=channels, name="Multi_Head_Attention")(x, x)
        
        attention_output = Reshape((height, width, channels), name="Reshape_to_Spatial")(attention_output)
        
        x = GlobalAveragePooling2D(name="Global_Avg_Pooling")(attention_output)
        x = Dense(512, activation='relu', name="FC_512")(x)
        x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, name="Batch_Normalization")(x)
        x = Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.016),
                  activity_regularizer=tf.keras.regularizers.l1(0.006),
                  bias_regularizer=tf.keras.regularizers.l1(0.006),
                  activation='relu', name="FC_256")(x)
        x = Dropout(0.45, seed=123, name="Dropout")(x)
        
        outputs = Dense(num_classes, activation='softmax', name="Output_Layer")(x)
        
        model = Model(inputs=inputs, outputs=outputs, name="Model_with_Attention")
        model.compile(optimizer=Adamax(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model

    input_shape = (224, 224, 3)
    model = create_model(input_shape, num_classes=15, learning_rate=0.0001)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    hist = model.fit(train_gen, epochs=50, callbacks=[early_stopping],
                     validation_data=valid_gen, verbose=1, shuffle=False)

    val_score = model.evaluate(valid_gen)

    val_accuracies.append(val_score[1])
    
    if val_score[1] > best_val_score:
        best_val_index = val_index
        best_val_score = val_score[1]
        best_histories.append(hist.history)

best_valid_df = df.iloc[best_val_index]

best_valid_gen = gen.flow_from_dataframe(best_valid_df, x_col='Picture Path', y_col='Label', target_size=img_size, 
                                         class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=batch_size)

preds = model.predict(best_valid_gen)
y_pred = np.argmax(preds, axis=1)

g_dict = best_valid_gen.class_indices
classes = list(g_dict.keys())

cm = confusion_matrix(best_valid_gen.classes, y_pred)

plt.figure(figsize=(15, 15))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
plt.title('Confusion Matrix - Best Validation Set')
plt.colorbar()

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')

plt.tight_layout()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

print("Classification Report for the Best Validation Set")
print(classification_report(best_valid_gen.classes, y_pred, target_names=classes))
