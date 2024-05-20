#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 21:34:23 2023

@author: alaric
"""

# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet121, EfficientNetB0
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_ResNet50
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_DenseNet121
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_input_EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import fbeta_score

# Define the data paths
base_dir = '/Users/alaric/Desktop/Deep Learning for Image & Video Processing (KEN4244)/aptos2019-blindness-detection'
train_dir = base_dir + '/train_images'
train_csv = base_dir + '/train.csv'
test_dir = base_dir + '/test_images'
test_csv = base_dir + '/test.csv'

# Load data
train_df = pd.read_csv(train_csv)
train_df['id_code'] = train_df['id_code'].astype(str) + '.png'
train_df['diagnosis'] = train_df['diagnosis'].astype(str)

# Split the data into training and validation sets
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Apply preprocessing to the validation DataFrame
val_df['id_code'] = val_df['id_code'].astype(str) + '.png'
val_df['diagnosis'] = val_df['diagnosis'].astype(str)

# Upsampling function to balance classes
def upsample_classes(df, class_col='diagnosis'):
    """
    Upsamples the minor classes in the dataframe to match the major class's count.
    
    Parameters:
    df (DataFrame): The DataFrame containing the training data.
    class_col (str): The name of the column containing class labels.
    
    Returns:
    DataFrame: An upsampled DataFrame.
    """
    max_size = df[class_col].value_counts().max()
    lst = [df]
    for class_index, group in df.groupby(class_col):
        lst.append(group.sample(max_size-len(group), replace=True))
    df_new = pd.concat(lst)
    return df_new

# Upsample the training DataFrame
upsampled_train_df = upsample_classes(train_df)

def create_generators(model_name, train_df, val_df, train_dir, val_dir, batch_size=32):
    """
    Creates image data generators for training and validation datasets.
    
    Parameters:
    model_name (str): The name of the model (to select the correct preprocessing).
    train_df (DataFrame): The training DataFrame.
    val_df (DataFrame): The validation DataFrame.
    train_dir (str): Directory path for training images.
    val_dir (str): Directory path for validation images.
    batch_size (int): The size of the batch.
    
    Returns:
    tuple: A tuple containing the train and validation generators.
    """
    # Select the appropriate preprocessing function and image size based on the model
    if model_name == 'VGG16':
        image_size = (224, 224)
        preprocess_input = preprocess_input_VGG16
    elif model_name == 'ResNet50':
        image_size = (224, 224)
        preprocess_input = preprocess_input_ResNet50
    elif model_name == 'DenseNet121':
        image_size = (224, 224)
        preprocess_input = preprocess_input_DenseNet121
    elif model_name == 'EfficientNetB0':
        image_size = (224, 224)
        preprocess_input = preprocess_input_EfficientNetB0
    else:
        raise ValueError("Unsupported model name")

    # Data augmentation configuration for training data
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # No augmentation for validation data, only preprocessing
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # Generator for training data
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=train_dir,
        x_col='id_code',
        y_col='diagnosis',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Generator for validation data
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=val_dir,
        x_col='id_code',
        y_col='diagnosis',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, val_generator

# Model creation and training
def create_and_train_model(model_name, train_generator, val_generator, epochs=10):
    """
    Creates, compiles, and trains a model based on the given model name.

    Parameters:
    model_name (str): The name of the model to use.
    train_generator (ImageDataGenerator): The generator for training data.
    val_generator (ImageDataGenerator): The generator for validation data.
    epochs (int): The number of epochs to train for.

    Returns:
    Model: The trained Keras model.
    dict: A dictionary containing training history.
    """
    if model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=train_generator.image_shape)
    elif model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=train_generator.image_shape)
    elif model_name == 'DenseNet121':
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=train_generator.image_shape)
    elif model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=train_generator.image_shape)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    for layer in base_model.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    num_classes = len(train_generator.class_indices)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
    model_checkpoint = ModelCheckpoint(filepath=f'model_{model_name}.h5', save_best_only=True, monitor='val_loss', mode='min')

    steps_per_epoch = max(train_generator.samples // train_generator.batch_size, 1)
    validation_steps = np.ceil(val_generator.samples / val_generator.batch_size)

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=[early_stopping, model_checkpoint]
    )

    return model, history.history

# Ensemble prediction
def ensemble_predict(models, val_generator):
    """
    Makes predictions by ensemble of models.

    Parameters:
    models (dict): A dictionary of trained models.
    val_generator (ImageDataGenerator): The generator for validation data.

    Returns:
    np.array: Array of ensemble predictions.
    """
    # Collect predictions from each model
    model_predictions = [model.predict(val_generator) for model in models.values()]

    # Calculate the ensemble predictions as the average of the individual model predictions
    ensemble_predictions = np.mean(np.array(model_predictions), axis=0)

    # Choose the class with the highest prediction for each sample
    final_predictions = np.argmax(ensemble_predictions, axis=1)
    return final_predictions

def calculate_balanced_f_score(y_true, y_pred):
    """
    Calculate the balanced F-score of the classifier.
    
    Parameters:
    y_true (np.array): True labels.
    y_pred (np.array): Predictions from the classifier.

    Returns:
    float: The balanced F-score of the classifier.
    """
    # Can adjust the beta parameter if a different F-score is needed
    return fbeta_score(y_true, y_pred, beta=1, average='weighted')

if __name__ == "__main__":
    # Load data
    train_df = pd.read_csv(train_csv)
    train_df['id_code'] = train_df['id_code'].astype(str) + '.png'
    train_df['diagnosis'] = train_df['diagnosis'].astype(str)

    # Split and upsample
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    upsampled_train_df = upsample_classes(train_df)

    # Define model names and initialize containers
    model_names = ['VGG16', 'ResNet50', 'DenseNet121', 'EfficientNetB0']
    trained_models = {}
    model_accuracies = {}
    model_f1_scores = {}
    f_scores = {}
    model_weights = {}
    batch_size = 32  # Define batch size

    # Create a single validation generator for all models
    _, val_gen = create_generators(model_names[0], upsampled_train_df, val_df, train_dir, train_dir, batch_size)

    # Train each model and collect their accuracies and F1 scores
    for model_name in model_names:
        print(f"Training {model_name}...")
        train_gen, _ = create_generators(model_name, upsampled_train_df, val_df, train_dir, train_dir, batch_size)
        model, history = create_and_train_model(model_name, train_gen, val_gen)
        trained_models[model_name] = model
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Train Accuracy')  # Corrected line
        plt.plot(history['val_accuracy'], label='Validation Accuracy')  # Corrected line
        plt.title(f'{model_name} Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
    
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Train Loss')  # Corrected line
        plt.plot(history['val_loss'], label='Validation Loss')  # Corrected line
        plt.title(f'{model_name} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
    
        plt.show()
    
        # Evaluate each model
        val_gen.reset()  # Reset validation generator
        steps = np.ceil(len(val_df) / batch_size)
        predictions = model.predict(val_gen, steps=steps)
        predicted_classes = np.argmax(predictions, axis=1)
        true_labels = val_gen.classes
        accuracy = accuracy_score(true_labels, predicted_classes)
        f1 = f1_score(true_labels, predicted_classes, average='weighted')
    
        # Store the results
        model_accuracies[model_name] = accuracy
        model_f1_scores[model_name] = f1
        print(f"{model_name} - Accuracy: {accuracy}, F1 Score: {f1}")
    
        # Calculate and store the balanced F-score for each model
        f_scores[model_name] = f1

    # Calculate weights for each classifier based on F-scores
    total_f_score = sum(f_scores.values())
    for model_name, f_score in f_scores.items():
        model_weights[model_name] = f_score / total_f_score

    # Make weighted ensemble predictions
    val_gen.reset()
    weighted_predictions = np.zeros((len(val_df), trained_models[model_names[0]].output_shape[1]))

    for model_name, model in trained_models.items():
        val_gen.reset()
        predictions = model.predict(val_gen, steps=steps)
        weighted_predictions += predictions * model_weights[model_name]

    # Obtain the final ensemble predictions
    ensemble_predictions = np.argmax(weighted_predictions, axis=1)

    # Evaluate the ensemble model
    ensemble_accuracy = accuracy_score(true_labels, ensemble_predictions)
    ensemble_f1_score = f1_score(true_labels, ensemble_predictions, average='weighted')

    # Add Ensemble Performance to dictionaries for plotting
    model_accuracies['Ensemble'] = ensemble_accuracy
    model_f1_scores['Ensemble'] = ensemble_f1_score

    # Display ensemble results
    print(f"Ensemble Model - Accuracy: {ensemble_accuracy}, F1 Score: {ensemble_f1_score}")

    # Plotting the accuracies and F1 scores
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(model_accuracies.keys(), model_accuracies.values(), align='center', alpha=0.7, color='b')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracies')
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    plt.bar(model_f1_scores.keys(), model_f1_scores.values(), align='center', alpha=0.7, color='g')
    plt.ylabel('F1 Score')
    plt.title('Model F1 Scores')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()
