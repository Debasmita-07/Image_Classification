🧠 CIFAR-10 Image Classifier with TensorFlow & OpenCV

This project loads a pre-trained image classification model trained on the CIFAR-10 dataset and uses it to classify new images.

📁 Dataset

CIFAR-10: Contains 60,000 32x32 color images in 10 classes, with 6,000 images per class.
Classes:
['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
🚀 How It Works

Loads the CIFAR-10 dataset from TensorFlow datasets.
Displays sample training images with their labels.
Loads a pre-trained .keras model (image_classifier.keras).
Reads a custom image, resizes and processes it.
Predicts the class of the image using the trained model.
📦 Requirements

pip install tensorflow matplotlib opencv-python
🧪 Usage

Train or Load Your Model
If not already trained, first train your model and save it:

model.save('image_classifier.keras')
Or use your already trained model like this:

model = models.load_model('image_classifier.keras')
Run the Prediction Code
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images / 255, testing_images / 255

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

model = models.load_model('image_classifier.keras')

img = cv.imread('path_to_your_image.jpg')
img = cv.resize(img, (32, 32))  # Resize to 32x32 for CIFAR model
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img)
plt.show()

prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction)
print(f'Prediction: {class_names[index]}')
📌 Notes

Images must be resized to 32x32 pixels to match CIFAR-10 input shape.
Make sure your .keras model is trained using the same preprocessing pipeline.
This classifier is only accurate on images similar to CIFAR-10 categories.
📸 Example Output

Prediction: Dog
Let me know if you'd like me to generate this as a downloadable README.md file or include training code too!
