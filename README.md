# Brain Metastasis Segmentation 
This project focuses on segmenting brain metastases in MRI images using deep learning. We implemented two models: Nested U-Net (U-Net++) and Attention U-Net. Additionally, we built a web application using FastAPI and Streamlit to visualize the segmentation results.

## Key Steps

### 1. Data Preprocessing
- Enhanced image contrast using CLAHE to make brain metastases more visible.
- Normalized images and applied data augmentation (e.g., rotations, flips) to improve model accuracy.

### 2. Models
- **Nested U-Net (U-Net++)**: Improves segmentation by using dense skip connections, helping to capture fine details.
- **Attention U-Net**: Adds attention gates to focus on important regions of the MRI images, improving detection of metastases.

### 3. Model Training and Evaluation
- Both models were trained using the Dice Score as the evaluation metric.
- **U-Net++** is better for complex structures, while **Attention U-Net** performs better with smaller lesions.

### 4. Web Application
- The app uses **FastAPI** for backend processing and **Streamlit** for the user interface.
- Users can upload MRI images and view segmentation results via a browser.

### 5. Running the App
- Clone the repository and install the dependencies.
- Run the FastAPI backend and Streamlit frontend.
- Open the app in a browser to upload images and see the results.

### 6. Challenges & Solutions
- **Small lesions**: Improved detection with CLAHE and Attention U-Net.
- **Data imbalance**: Attention U-Net helps focus on rare, smaller metastases.

## Conclusion
We built a robust solution using deep learning to segment brain metastases and created a user-friendly web application for visualizing the results.
