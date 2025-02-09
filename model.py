import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout,
    BatchNormalization, GaussianNoise, Input,
    MultiHeadAttention, Reshape
)
from tensorflow.keras.optimizers import Adamax

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
    x = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.001, name="Batch_Normalization")(x)
    x = Dense(256, kernel_regularizer=regularizers.l2(0.016), activity_regularizer=regularizers.l1(0.006),
              bias_regularizer=regularizers.l1(0.006), activation='relu', name="FC_256")(x)
    x = Dropout(0.45, seed=123, name="Dropout")(x)
    
    outputs = Dense(num_classes, activation='softmax', name="Output_Layer")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="Model_with_Attention")
    model.compile(optimizer=Adamax(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model