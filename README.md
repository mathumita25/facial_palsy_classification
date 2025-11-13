# Automated Facial Palsy Severity Classification Using Geometric Angle Features and Random Forest

## Abstract
This project presents an automated facial palsy detection and severity classification system using geometric angle features extracted from facial landmarks. We propose a Random Forest classifier trained on three key angular measurements: eyebrow angle, eye angle, and mouth angle. The system achieves 90.12% accuracy in classifying facial images into four severity categories: Normal, Mild, Moderate, and Severe. Our approach combines traditional facial landmark detection with machine learning to provide objective, reproducible severity assessments for clinical applications.

## Team Members
- **Mathumita S** - [22MIA1045]
- **Sai Lakshmini R** - [22MIA1042]

## Base Paper Reference
**"Facial Landmark-Based Emotion Recognition via Directed Graph Neural Network"**  
*Quang Tran Ngoc, Seunghyun Lee, and Byung Cheol Song*  
Electronics 2020, 9(5), 764; https://doi.org/10.3390/electronics9050764

## Tools and Libraries Used
- **Platform**: Google Colab
- **Programming Language**: Python 3.8+
- **Computer Vision**: OpenCV, dlib
- **Machine Learning**: scikit-learn, Random Forest
- **Data Processing**: NumPy, pandas
- **Visualization**: Matplotlib, seaborn

## Dataset Information

### Original Unbalanced Dataset
- **Total Images**: 18,000 records in CSV
- **YFP Facial Palsy Images**: 14,000 images (77.8%)
- **CelebA Normal Images**: 4,000 images (22.2%)
- **Source Links**:
  - YFP Dataset: [https://www.kaggle.com/datasets/dohaeid/yfp-dataset-updated]
  - CelebA Dataset: [https://www.kaggle.com/datasets/jessicali9530/celeba-dataset]

### Balanced Training Dataset
- **Total records used for training the random forest model**: 8,000 records (after balancing)
- **Classes**: 4 severity levels ( made from 2,000 images each)
  - Class 0: Normal
  - Class 1: Mild
  - Class 2: Moderate
  - Class 3: Severe

## Notebooks Description

### 1. Dataset Creation & Feature Extraction
**File**: `1_palsy_features_dataset_creation_analysis.ipynb`

**Purpose**: Processes raw images and extracts geometric features from unbalanced dataset
- Loads 18K raw images (14K YFP + 4K CelebA)
- Performs face detection using OpenCV
- Extracts 68 facial landmarks using dlib
- Computes three geometric angle features to create a dataset(csv file) that contains:
  - Eyebrow angle (asymmetry measurement)
  - Eye angle (asymmetry measurement) 
  - Mouth angle (asymmetry measurement)
- Generates exploratory data analysis and visualizations
- Output: `facial_angle_analysis_results.csv` (18k records)

### 2. Model Training & Prediction
**File**: `3_predicting_palsy_using_created_dataset.ipynb`

**Purpose**: Trains and evaluates the Random Forest model on balanced dataset
- Load the facial_angle_analysis_results.csv and create a balanced dataset (8K records)
- Trains Random Forest classifier with 200 trees
- Evaluates model performance:
  - Overall Accuracy: 90.12%
  - Precision: 89-96% across classes
  - Recall: 86-97% across classes
- Generates confusion matrix and feature importance analysis
- Saves trained model: `random_forest_model.pkl`
- Provides inference on new images

## Steps to Execute in Google Colab

### Method 1: Direct Upload to Colab
1. Upload both notebooks to Google Colab
2. Run them in order:
   - First: `1_palsy_features_dataset_creation_analysis.ipynb`
   - Then: `2_predicting_palsy_using_created_dataset.ipynb`
3. Upload required data files to Colab storage

### Method 2: Clone Repository in Colab
```python
# Run in Colab cell
!git clone https://github.com/yourusername/facial-palsy-classification.git
%cd facial_palsy_classification

# Install dependencies
!pip install -r requirements.txt

# Download dlib shape predictor
!wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
!bunzip2 shape_predictor_68_face_landmarks.dat.bz2
!mv shape_predictor_68_face_landmarks.dat models/

# Run the first notebook (feature extraction and balancing)
!jupyter nbconvert --to notebook --execute notebooks/1_palsy_features_dataset_creation_analysis.ipynb

# Run the second notebook (model training and evaluation)
!jupyter nbconvert --to notebook --execute notebooks/3_predicting_palsy_using_created_dataset.ipynb


Output screenshots

![facial_palsy_classification](output%20results.png)


