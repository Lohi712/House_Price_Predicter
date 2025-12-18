# Bangalore House Price Predictor
An end-to-end Machine Learning web application that predicts real estate prices in Bangalore. It uses an advanced Log-Transformed CatBoost Regression pipeline to handle non-linear price trends and achieves an 86.3% R^2 Score.
The application features a Python Flask backend and a modern, responsive frontend with a Glassmorphism UI design.
<img width="1912" height="932" alt="image" src="https://github.com/user-attachments/assets/c3c8ac64-138b-4068-b4de-e93bd27901e4" />

## Model Performance & Methodology
Achieving high accuracy on real-world housing data is difficult due to outliers and non-linear pricing (e.g., luxury properties). I experimented with multiple algorithms before finding the best working model.

| Model                          | Accuracy (R² Score) | Status         |
|--------------------------------|---------------------|--------------- |
| Linear Regression              | 71.36%              | Baseline       |
| Lasso Regression (L1)          | 70.58%              | Underfitting   |
| Ridge Regression               | 71.36%              | Not much change|
| Random Forest                  | 77.54%              | Good           |
| Voting Regressor (Ensemble)    | 77.69%              | Better         |
| CatBoost + Log Transform       | 86.27%              | Champion       |


### Data Engineering:
Handled missing values and inconsistent area units (ranges vs exact numbers).
Performed Dimensionality Reduction on 1000+ location categories using "Other" grouping.
Engineered new features like Price_Per_Sqft for outlier detection.

### Advanced Modeling:
Used Log-Transformation on the target variable (Price) to normalize the skewed distribution of real estate prices.
Implemented CatBoostRegressor, which naturally handles categorical data better than OneHotEncoding.

### Deployment:
Model serialized using pickle.
Served via a Flask REST API.
Frontend built with HTML/CSS/JavaScript (jQuery) for real-time interaction.

## Tech Stack
Language: Python
Machine Learning: Scikit-Learn, CatBoost, XGBoost, NumPy, Pandas
Backend: Flask (Python Microframework)
Frontend: HTML5, CSS3 (Glassmorphism), JavaScript (jQuery)
IDE: VS Code

Frontend built with HTML/CSS/JavaScript (jQuery) for real-time interaction.

# How to Run Locally:
1. Clone the repository (or download the files): git clone https://github.com/your-username/Bangalore-House-Price-Predictor.git
cd Bangalore-House-Price-Predictor
2. Install Dependencies: pip install flask scikit-learn pandas numpy catboost
3. Run the Server: python app.py
4. Access the App: Open your browser and go to: http://127.0.0.1:5000
