import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Parameters
n_packets = 200  # Example, for raw data reference if needed (can be ignored if irrelevant)


def load_data_from_txt(file_path, label_path=None):
    """
    Load pose signatures and optional labels from a txt file.

    Args:
        file_path (str): Path to the txt file containing 2D NumPy array (features).
        label_path (str): Path to the txt file containing labels (if available).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Features (X) and labels (y).
    """
    # Load signatures from the txt file
    X = np.loadtxt(file_path, delimiter=',')

    # Optionally load labels
    if label_path:
        labels = np.loadtxt(label_path, dtype=str)
        y = np.array([''.join(row) for row in labels])

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    else:
        y = None  # No labels available

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)


    return X, y


# Build a CNN model for classification
def build_simple_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.5),  # Add dropout to avoid overfitting
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Main function
if __name__ == "__main__":
    # File paths
    signatures_file = "signatures.txt"  # Replace with the path to your signatures file
    labels_file = "labels.txt"  # Replace with the path to your labels file (optional)

    # Load data
    X, y = load_data_from_txt(signatures_file, labels_file)



    if y is None:
        print("Labels not provided. Please include a label file for supervised training.")
    else:
        # Perform stratified random split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )

        print("Train and test singnatures used: ", X_train.shape, X_test.shape)
        np.savetxt("y_train.txt", y_train, fmt='%s')
        np.savetxt("y_test.txt", y_test, fmt='%s')

        # Build and train the model
        input_shape = X_train.shape[1:]
        num_classes = len(np.unique(y))
        model = build_simple_model(input_shape, num_classes)

        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32)

        # Evaluate the model
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

        print("Train Accuracy:", model.evaluate(X_train, y_train, verbose=0)[1])
        print("Test Accuracy:", model.evaluate(X_test, y_test, verbose=0)[1])
