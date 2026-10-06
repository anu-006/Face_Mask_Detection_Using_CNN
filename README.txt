FACE MASK DETECTION USING CNN
==============================

1. PROJECT OVERVIEW
-------------------
This project uses a Convolutional Neural Network (CNN) to detect whether a
person is wearing a face mask or not.

The model classifies an input face image into two classes:
    1. with_mask
    2. without_mask

The complete workflow is available in:
    Face_Mask_Detection.ipynb

2. TECHNOLOGIES USED
--------------------
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

3. PROJECT STRUCTURE
--------------------
The dataset does NOT need to be uploaded to the project repository.
The notebook already contains the Kaggle dataset link/access steps.

Face-Mask-Detection/
|
|-- Face_Mask_Detection.ipynb
|-- requirements.txt
|-- README.txt


4. DATASET
----------
The project uses a face-mask image dataset from Kaggle.
You do not need to manually include the dataset in this project folder.
The Kaggle dataset link/access information is already provided inside the
notebook. Follow those steps before running the model training cells.

The dataset contains two image classes:
    with_mask
    without_mask

5. HOW THE PROJECT WORKS
------------------------
Kaggle Dataset -> Load/access dataset -> Image preprocessing -> Resize to
256 x 256 -> Normalize pixel values -> CNN -> Training -> Validation ->
Testing -> Mask / No Mask prediction

6. MODEL
--------
A Convolutional Neural Network (CNN) is used for binary image classification.
The CNN learns visual features and classifies an image as either with_mask
or without_mask.

7. INSTALLATION
---------------
Use Python 3.10 or 3.11.

Create a virtual environment (recommended):

    python -m venv venv
    venv\\Scripts\\activate

Install the required libraries:

    pip install -r requirements.txt

8. RUNNING THE PROJECT
----------------------
1. Open the project folder in VS Code, Jupyter Notebook, or JupyterLab.
2. Open Face_Mask_Detection.ipynb.
3. Follow the Kaggle dataset access/download instructions in the notebook.
4. Run the notebook cells from top to bottom.
5. Use the prediction section to test an image.

9. IMAGE PREPROCESSING
----------------------
Images are resized to 256 x 256 pixels and normalized to the range 0-1.

Example:
    image = cv2.resize(image, (256, 256))
    image = image.reshape(1, 256, 256, 3)
    image = image / 255.0

Training, validation, testing, and individual prediction images should use
the same preprocessing steps.

10. MODEL EVALUATION
--------------------
The notebook uses evaluation metrics and visualizations such as:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

11. COMMON PROBLEMS
-------------------
Problem: ModuleNotFoundError
Solution:
    pip install -r requirements.txt

Problem: Dataset not found
Solution: Check the Kaggle dataset access/download section in the notebook
and make sure the dataset is available at the path expected by the notebook.

Problem: Poor or unexpected predictions
Solution: Make sure prediction images use the same preprocessing as the
training data, especially resizing to 256 x 256 and normalization by 255.0.

12. PROJECT PURPOSE
-------------------
This is a beginner-friendly deep learning project for understanding image
classification using a Convolutional Neural Network (CNN). It demonstrates
how a CNN can be trained and evaluated to classify face images as either
wearing a mask or not wearing a mask.
