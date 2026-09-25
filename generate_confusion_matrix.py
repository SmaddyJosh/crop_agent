import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import argparse

def main():
    parser = argparse.ArgumentParser(description='Generate a confusion matrix for the model.')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test'],
                        help='Dataset split to evaluate on (train, valid, test)')
    args = parser.parse_args()

    model_path = 'crop_simple.keras'
    data_dir = 'datasets/all_data'
    img_size = (180, 180)
    batch_size = 32

    # Load model
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    
    # Load dataset
    print(f"Loading test dataset using validation_split=0.3 and seed=123...")
    val_test_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.3,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )
    
    # split the 30% val_test_ds into 15% validation and 15% test
    val_batches = tf.data.experimental.cardinality(val_test_ds)
    test_ds = val_test_ds.take(val_batches // 2) # Note: this must precisely match model.py's split logic
    
    class_names = val_test_ds.class_names

    # Get predictions
    print("Generating predictions...")
    y_pred = []
    y_true = []
    
    for x, y in test_ds:
        preds = model.predict(x, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(y.numpy())
        
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # Compute confusion matrix
    cm = tf.math.confusion_matrix(y_true, y_pred).numpy()

    # Plot confusion matrix
    print("Plotting confusion matrix...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save image
    output_path = 'confusion_matrix.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Confusion matrix saved to {output_path}")

if __name__ == '__main__':
    main()
