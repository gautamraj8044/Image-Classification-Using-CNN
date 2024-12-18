# CIFAR-10 Image Classification with Convolutional Neural Network (CNN)

## Overview
This project implements a **Convolutional Neural Network (CNN)** to classify images from the **CIFAR-10** dataset. The dataset contains 60,000 32x32 color images across 10 categories, with 50,000 images for training and 10,000 for testing. Each category includes 6,000 images, making it an excellent benchmark for image classification tasks.



## Dependencies
Ensure you have the following installed:

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib

You can install the required packages using:

```bash
pip install tensorflow numpy pandas matplotlib
```

## Usage

### Dataset
The CIFAR-10 dataset can be loaded directly from TensorFlow:

```python
from tensorflow.keras import datasets
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
```

Ensure the dataset is properly loaded and preprocessed before training.

### Running the Script
To run the project, navigate to the `src` directory and execute the main script:

```bash
python Image_Classification_using_CNN.ipynb
```

### Main Components
1. **Data Loading and Preprocessing**:
   - The CIFAR-10 dataset is loaded and normalized to a range of [0, 1].
   - Labels are flattened for compatibility with sparse categorical cross-entropy loss.

2. **Model Definition**:
   - **Convolutional Layers**: Extract features using filters of increasing size (32, 64, 128).
   - **Max Pooling Layers**: Reduce spatial dimensions to prevent overfitting.
   - **Flatten Layer**: Converts feature maps into 1D vectors.
   - **Dense Layers**:
     - A hidden layer with dropout for regularization.
     - An output layer with softmax activation for multi-class classification.

3. **Model Compilation**:
   - Optimizer: **Adam**
   - Loss: **Sparse Categorical Cross-Entropy**
   - Metric: **Accuracy**

4. **Training**:
   - Trained for 20 epochs with validation on the test set.

5. **Evaluation**:
   - Final test loss and accuracy are printed after training.

### Results
After training, the model outputs its performance on the test set, including:

- **Test Loss**
- **Test Accuracy**

## Future Improvements
To further enhance performance, consider:
- Implementing **data augmentation** to reduce overfitting.
- Using **transfer learning** with pre-trained models like VGG or ResNet.
- Experimenting with different hyperparameters (e.g., learning rate, batch size, epochs).
- Adding **callbacks** such as early stopping and model checkpointing.

## Contributing
Pull requests are welcome! For major changes, please open an issue to discuss proposed modifications before contributing.

## Contact
For any questions or feedback, feel free to reach out:

**Gautam Raj**  
[gautamraj8044@gmail.com](mailto:gautamraj8044@gmail.com)

---

This `README.md` provides a clear and structured overview of the project, making it easy for others to understand, set up, and run.
``` 

