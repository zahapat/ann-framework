import os
import six
import gzip
import struct
import numpy as np
import matplotlib.pyplot as plt


# Download a specific fashion MNIST data file
def download_fashion_mnist_dataset(file_name):
    if not os.path.exists(f"{file_name}"):
        print(f"PY: Downloading file: {file_name}...")
        root_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com'
        full_url_path = root_url+'/'+file_name
        downloaded_file_path, http_msg = six.moves.urllib.request.urlretrieve(
            url=full_url_path,
            filename=file_name)
        print(f"PY: Downloading file: {file_name} done.")
    else:
        print(f"PY: File: {file_name} has already been downloaded.")
        downloaded_file_path = file_name

    with gzip.GzipFile(downloaded_file_path) as f:
        downloaded_data = f.read()

    return downloaded_data


# Extract numerical values out of the binary labels files and change shape for ease of use
def parse_fashion_mnist_labels(downloaded_data):
    print(f"PY: Parsing labels file...")
    target_magic = 2049
    magic, examples = struct.unpack_from(">2i", downloaded_data)
    assert magic == target_magic
    labels = np.frombuffer(downloaded_data[8:], dtype=np.uint8)
    print(f"PY: Parsing labels file done.")
    return labels


# Extract numerical values out of the binary images files and change shape for ease of use
def parse_fashion_mnist_images(downloaded_data):
    print(f"PY: Parsing images file...")
    target_magic = 2051
    magic, number, rows, cols = struct.unpack_from(">4i", downloaded_data)
    assert magic == target_magic
    images = np.frombuffer(downloaded_data[16:], dtype=np.uint8).astype(np.float32)
    images = np.reshape(images, (number, rows, cols))
    print(f"PY: Parsing images file done.")
    return images


# Download all fashion MNIST datasets
def download_all_fashion_mnist_datasets():

    # Training Set Images: 26 MBytes, MD5 Checksum: 8d4fb7e6c68d591d4c3dfef9ec88bf0d
    training_images = download_fashion_mnist_dataset('train-images-idx3-ubyte.gz')

    # Training Set Labels: 29 KBytes, MD5 Checksum: 25c81989df183df01b3e8a0aad5dffbe
    training_labels = download_fashion_mnist_dataset('train-labels-idx1-ubyte.gz')

    # Test Set Images: 4.3 MBytes, MD5 Checksum: bef4ecab320f06d8554ea6380940ec79
    test_images = download_fashion_mnist_dataset('t10k-images-idx3-ubyte.gz')

    # Test Set Labels: 5.1 KBytes, MD5 Checksum: bb300cfdad3c16e7a12a480ee83cd310
    test_labels = download_fashion_mnist_dataset('t10k-labels-idx1-ubyte.gz')

    return training_images, training_labels, test_images, test_labels

# Parse all images and labels
def parse_all_fashion_mnist_datasets(training_images, training_labels, test_images, test_labels):
    X = parse_fashion_mnist_images(training_images)
    y = parse_fashion_mnist_labels(training_labels)
    X_test = parse_fashion_mnist_images(test_images)
    y_test = parse_fashion_mnist_labels(test_labels)
    return X, y, X_test, y_test

# Fashion MNIST dataset (train + test data)
def load_fashion_mnist_dataset():
    training_images, training_labels, test_images, test_labels = \
        download_all_fashion_mnist_datasets()
    X, y, X_test, y_test = parse_all_fashion_mnist_datasets(
        training_images, training_labels, test_images, test_labels
    )
    return X, y, X_test, y_test



# Pre-processing of the Fashion MNIST dataset
def dataset_mnist_preprocess(X, y, X_test, y_test):
    # Re-scale
    X = (X.astype(np.float32) - 127.5) / 127.5
    X_test = (X_test.astype(np.float32) - 127.5) / 127.5

    # Re-shape
    X = X.reshape(X.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Shuffle all images and lables the same way
    keys = np.array(range(X.shape[0]))
    np.random.shuffle(keys)
    X = X[keys]
    y = y[keys]

    return X, y, X_test, y_test


# Fashion MNIST dataset (train + test data)
def dataset_mnist_create(display_image_number=None):
    # Get data from online dataset source
    X, y, X_test, y_test = load_fashion_mnist_dataset()

    # Perform pre-processing on the train and test data
    X, y, X_test, y_test = dataset_mnist_preprocess(X, y, X_test, y_test)

    # Show an image if requested before pre-processing
    if display_image_number is not None:
        plt.imshow((X[display_image_number].reshape(28, 28))) # Reshape image from vector to 28x28 matrix
        plt.show()
        print(f'y[{display_image_number}] = {y[display_image_number]}')
        print(f'X[{display_image_number}]:')
        print(X[display_image_number])

    # And return all the data
    return X, y, X_test, y_test
