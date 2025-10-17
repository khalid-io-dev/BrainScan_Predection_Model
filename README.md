Brain Tumor Classifier - README
===============================

What It Does
------------

This project is a machine learning application designed to assist in the early detection of brain tumors by analyzing MRI images using a Convolutional Neural Network (CNN). Developed as part of an initiative by BrainScan AI, a Moroccan startup focused on AI-assisted medical imaging, the application aims to enhance diagnostic accuracy and speed for medical professionals. It includes:

*   **Data Preprocessing**: Loads, validates, resizes, and normalizes images, ensuring data quality and consistency.
    
*   **Model Training**: Builds and trains a CNN to recognize tumor patterns, with options for hyperparameter tuning and regularization.
    
*   **Evaluation**: Assesses model performance with metrics like accuracy, confusion matrices, and classification reports, including visualizations.
    
*   **Deployment**: Provides a Streamlit web interface for real-time predictions on uploaded or local images, leveraging a saved model.
    

Installation
------------

### Prerequisites

Ensure you have Python 3.8 or higher installed on your system. You can download it from [python.org](https://www.python.org/downloads/).

### Dependencies

Install the required Python packages using pip. Run the following command in your terminal or command prompt:

bash ```
pip install -r requirements.txt   `

```

If you don't have a requirements.txt file yet, create one with the following content and run the command above:

text

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   streamlit==1.30.0  tensorflow==2.15.0  opencv-python==4.9.0.80  numpy==1.26.2  Pillow==10.2.0  scikit-learn==1.3.2  matplotlib==3.8.2  seaborn==0.13.2  pandas==2.2.0   `

#### Notes on Dependencies

*   **TensorFlow**: Used for building and training the CNN model.
    
*   **OpenCV**: Handles image reading, resizing, and preprocessing.
    
*   **Streamlit**: Creates the interactive web interface.
    
*   **NumPy**: Manages array operations for image data.
    
*   **Pillow**: Supports image processing for the Streamlit app.
    
*   **Scikit-learn**: Provides tools for label encoding and evaluation metrics.
    
*   **Matplotlib/Seaborn**: Generates plots and confusion matrices.
    
*   **Pandas**: Enhances data visualization in the Streamlit app.
    

### Additional System Requirements

*   **Graphviz** (optional): For plot\_model() to generate model architecture diagrams. Install via:
    
    *   Windows: Download from [Graphviz website](https://www.graphviz.org/download/) and add to PATH.
        
    *   Linux/Mac: sudo apt-get install graphviz or brew install graphviz.
        
*   Ensure you have write permissions in the project directory for saving models and figures.
    

How to Run
----------

### 1\. Prepare the Dataset

*   Place your MRI image dataset in the ./Data\_split directory with subfolders train, val, and test, each containing class-specific subfolders (e.g., glioma, meningioma, notumor, pituitary).
    
*   Ensure images are in supported formats (.jpg, .jpeg, .png, .bmp).
    

### 2\. Train the Model

*   cd C:\\Users\\ADMIN\\Desktop\\BRIEF-3
    
*   python prepare\_data.py
    
*   python train\_cnn.py
    
*   python train\_and\_evaluate\_cell.py
    
*   python save\_and\_predict.py
    

### 3\. Launch the Streamlit App

*   streamlit run app.py
    
*   Open your web browser and go to the URL provided (e.g., http://localhost:8501).
    
*   Upload an MRI image or enter a local path, then click "Predict" to see the classification results.
    

### 4\. View Results

*   The app displays the predicted tumor class, confidence probabilities, and an input image preview.
    
*   Check the "Model Performance Overview" expander for training metrics, plots, and example predictions (ensure figures/loss\_acc\_plot1.png and figures/loss\_acc\_plot2.png are saved from the training process).
    

Additional Notes
----------------

*   **Model Files**: Ensure a trained model (e.g., models/best\_cnn.h5) and label file (models/label\_classes.npy) are present, or the app will attempt to infer classes from the dataset.
    
*   **Troubleshooting**: If errors occur (e.g., model loading fails), verify file paths and TensorFlow installation. Check logs for details.
    
*   **Customization**: Adjust IMG\_SIZE, BATCH\_SIZE, or other hyperparameters in the scripts as needed for your dataset.
    

This setup enables you to train, evaluate, and deploy a brain tumor classifier with an intuitive interface for medical use.
