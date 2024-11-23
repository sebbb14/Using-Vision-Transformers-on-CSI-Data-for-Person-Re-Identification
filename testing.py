import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def randomize_data(X, y):
    # Combine X (signatures) and y (labels) for consistent shuffling
    combined = list(zip(X, y))
    np.random.shuffle(combined)  # Randomly shuffle the combined list

    # Unpack the shuffled data back into X and y
    X_shuffled, y_shuffled = zip(*combined)
    return np.array(X_shuffled), np.array(y_shuffled)


def split_data_balanced(X, y, train_ratio=0.8):
    unique_labels = np.unique(y)
    X_train, X_test, y_train, y_test = [], [], [], []

    for label in unique_labels:
        # Extract indices of all samples for the current label
        label_indices = np.where(y == label)[0]
        label_X = X[label_indices]
        label_y = y[label_indices]

        # Split data for the current label
        train_X, test_X, train_y, test_y = train_test_split(
            label_X, label_y, train_size=train_ratio, random_state=42
        )

        # Append the results
        X_train.extend(train_X)
        X_test.extend(test_X)
        y_train.extend(train_y)
        y_test.extend(test_y)

    # Convert to numpy arrays
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


def load_data_from_txt(file_path, label_path=None):

    # Load signatures from the txt file
    X = np.loadtxt(file_path, delimiter=',')

    # Optionally load labels
    if label_path:
        labels = np.loadtxt(label_path, dtype=str)

        # convert the labels into a 1d numpy array
        y = np.array([''.join(row) for row in labels])

        label_encoder = LabelEncoder() # converts string labels name into integers
        y = label_encoder.fit_transform(y)

    else:
        y = None  # No labels available

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)


    return X, y


def model_cnn(input_shape, num_classes):
    model = models.Sequential([
        # Flatten the input data (e.g., if using image-like data)
        layers.Flatten(input_shape=input_shape),

        # Fully connected layer
        layers.Dense(64, activation='relu'),


        # Output layer with softmax for classification
        layers.Dense(num_classes, activation='softmax')


    ])

    # Compile the model with basic settings
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

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

        X_train, X_test, y_train, y_test = split_data_balanced(X, y)
        X_train, y_train = randomize_data(X_train, y_train)
        X_test, y_test = randomize_data(X_test, y_test)


        np.savetxt("y_train.txt", y_train, fmt='%s')
        np.savetxt("y_test.txt", y_test, fmt='%s')

        # Build and train the model
        input_shape = X_train.shape[1:]
        num_classes = len(np.unique(y)) # total number of labels unique()
        model = model_cnn(X_train.shape[1:], len(np.unique(y)))


        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20 , batch_size=32)

        # Evaluate the model
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

        print("Train Accuracy:", model.evaluate(X_train, y_train, verbose=0)[1])
        print("Test Accuracy:", model.evaluate(X_test, y_test, verbose=0)[1])
