# Face Mask Detection Using CNN

A beginner-friendly computer vision project that uses a Convolutional Neural Network (CNN) to classify faces as **Mask** or **No Mask**. The project includes a training notebook and a webcam application that displays the prediction on screen.

## Project Highlights

- Trains a CNN for binary image classification.
- Detects faces in a live webcam feed using OpenCV's Haar Cascade detector.
- Displays a predicted class and a prediction score.
- Uses horizontal flipping for a mirror-style webcam view.
- Includes data augmentation to help the model learn from varied training images.
- Evaluates the model using classification metrics and visualizations in the notebook.

## Problem Statement

Checking whether people are wearing face masks can be repetitive when done manually. This project explores how computer vision and deep learning can automate basic mask/no-mask classification from face images.

## Solution Overview

The application first detects a face in the webcam frame. It crops the detected face, resizes it to **256 × 256** pixels, converts the image to RGB, and scales pixel values to the **0–1** range. The trained CNN then predicts whether the face belongs to the `with_mask` or `without_mask` class. OpenCV draws a coloured rectangle and prediction label around the face.

## Technologies Used

- **Python** — application and model code
- **TensorFlow / Keras** — CNN training and inference
- **OpenCV** — webcam access, face detection, and on-screen labels
- **NumPy** — image array processing
- **Jupyter Notebook / Google Colab** — model training and evaluation
- **Matplotlib, Seaborn, Scikit-learn** — visualizations and evaluation tools used in the notebook

## CNN Architecture (Short Explanation)

The model is a custom Sequential CNN built for two-class image classification:

1. **Input:** RGB image of size `256 × 256 × 3`.
2. **Convolution block 1:** `Conv2D (32 filters, 3 × 3, ReLU)` → Batch Normalization → `MaxPooling2D (2 × 2)`.
3. **Convolution block 2:** `Conv2D (64 filters, 3 × 3, ReLU)` → Batch Normalization → `MaxPooling2D (2 × 2)`.
4. **Convolution block 3:** `Conv2D (128 filters, 3 × 3, ReLU)` → Batch Normalization → `MaxPooling2D (2 × 2)`.
5. **Flatten:** Converts learned feature maps into a one-dimensional vector.
6. **Dense layers:** 128 → 64 → 32 neurons, using ReLU activation and Dropout (`0.1`) after each dense layer.
7. **Output:** One neuron with sigmoid activation to produce a binary classification score.

The notebook also uses training-time data augmentation: horizontal flipping, small rotations, zooming, and contrast changes. Batch Normalization helps stabilize training, while Dropout helps reduce overfitting. This is a custom CNN; it is not a pretrained model such as VGG or ResNet.

### Model Performance

The Face Mask Detection model was evaluated using training, validation, and test datasets.

| Metric | Result |
|---|---:|
| Training Accuracy | 91.62% |
| Validation Accuracy | 88.91% |
| Test Accuracy | 87.86% |
| Training Loss | 0.2173 |
| Validation Loss | 0.2938 |
| Test Loss | 0.2737 |

The model achieved **87.86% test accuracy**, demonstrating its ability to classify face-mask images on unseen test data. The difference between training and test accuracy indicates that there is still room to improve generalization.

*Note: Actual performance may vary depending on image quality, lighting, face angle, and the test dataset.*

## Dataset

The project uses the **Face Mask Detection** dataset by Omkar Gurav on Kaggle:

**Dataset link:** https://www.kaggle.com/datasets/omkargurav/face-mask-dataset

The dataset has two folders/classes:

- `with_mask`
- `without_mask`

### How to get the dataset

**Option A — Use the notebook in Google Colab (recommended):**

1. Open the dataset link above and sign in to Kaggle.
2. In Google Colab, open `Face_Mask_Detection_2.ipynb`.
3. Configure Kaggle API credentials in Colab if required. You can create/download your Kaggle API token (`kaggle.json`) from your Kaggle account settings. Keep this token private; never commit it to GitHub.
4. Run the notebook's dataset download cell:
   ```python
   !kaggle datasets download -d omkargurav/face-mask-dataset
   ```
5. Run the extraction cell and confirm that the images are available under:
   `/content/face-mask-dataset/data/`
6. Continue running the notebook cells in order.

**Option B — Download manually:**

1. Open the Kaggle dataset link and download the dataset ZIP file.
2. Extract it.
3. Make sure the extracted folder contains `data/with_mask/` and `data/without_mask/`.
4. If running the notebook locally, update its dataset path to the location where you extracted the files.

The dataset is not included in this repository. Follow Kaggle's current access and usage terms when downloading or using it.

## Project Structure

```text
Face-Mask-Detection/
├── Face_Mask_Detection.ipynb   # Training, augmentation, and evaluation
├── detect.py                     # Webcam-based detection script
├── mask_model.keras              # Saved trained CNN model
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

Filenames in your repository may differ. If your script is named `detect_2.py`, use that filename in the run command below. Keep `mask_model.keras` in the same folder as the detection script, unless you change the model path in the code.

## Setup

Python **3.11** is recommended for the listed dependency versions.

### Windows (PowerShell)

Open a terminal in the project folder and run:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks environment activation, use the Python executable inside `.venv` directly to install the packages:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the Webcam Application

1. Ensure the trained model file `mask_model.keras` is in the project folder.
2. Close other applications that may be using the webcam.
3. Activate the virtual environment.
4. Run the script:

   ```powershell
   python detect.py
   ```

   If your file is named `detect_2.py`, run `python detect_2.py` instead.
5. The webcam window displays the detected face, class label, and prediction score.
6. Press **Q** to close the application.

If the webcam does not open, check **Windows Settings → Privacy & security → Camera** and allow camera access for desktop apps. You can also try another camera index in the script, such as `cv2.VideoCapture(1)`.

## Image Preprocessing and Prediction Score

The webcam script uses the following preprocessing steps to match the training setup:

- Resize the face crop to `256 × 256` pixels.
- Convert OpenCV's BGR image to RGB.
- Convert pixel values to floating point and divide by `255.0`.
- Add a batch dimension before passing the image to the model.

The notebook's class order is:

- `0 = with_mask`
- `1 = without_mask`

The sigmoid output is therefore interpreted as the score for `without_mask`. A score of `0.5` or above is displayed as **No Mask**; a lower score is displayed as **Mask**. The displayed score is the model's output-based estimate, not a guarantee that the prediction is correct or a measure of overall model accuracy.

## Evaluation

The notebook includes model evaluation and visualizations such as accuracy, precision, recall, F1-score, and a confusion matrix. Add your measured results here after running the notebook if you want to report specific performance numbers. No accuracy figure is claimed in this README.

## Limitations

- Predictions can be affected by lighting, camera quality, face angle, occlusion, and distance from the camera.
- The model predicts only the two classes represented in the training dataset.
- A high prediction score does not guarantee a correct result.
- This is an educational project and should not be treated as a safety-critical monitoring system.

## Future Improvements

- Test the model on more varied real-world images.
- Add a performance report with measured evaluation results.
- Improve face detection and handle multiple camera conditions.
- Explore model optimization for faster inference on low-powered devices.

## Author / Contact

Add your name, GitHub profile, LinkedIn profile, and contact details here before publishing your portfolio repository.

---

*Built as a learning project to practise deep learning, image preprocessing, computer vision, model evaluation, and Python application development.*
