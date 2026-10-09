# 🌾 Crop Recommendation System

A machine learning-based desktop application that recommends suitable crops based on soil nutrient levels and environmental conditions. It uses a **Random Forest Classifier** trained on crop and soil data, with a simple **Tkinter** interface for entering values and viewing predictions.

---

## 🌟 Features

- 🌱 **Crop prediction** based on soil and climate parameters
- 🧪 Input controls for Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, soil pH, and rainfall
- 🎯 Displays the predicted crop with a confidence score
- 🌾 Shows alternative crop predictions
- 📊 Visual comparison between entered values and the predicted crop's average values
- 📈 Visualizes the top five crop predictions and their confidence scores
- 🤖 Trains a Random Forest model when the application starts
- 🖥️ User-friendly desktop interface built with Tkinter

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Programming Language** | Python |
| **Desktop GUI** | Tkinter |
| **Machine Learning** | scikit-learn (`RandomForestClassifier`) |
| **Data Processing** | pandas, NumPy |
| **Feature Scaling** | `StandardScaler` |
| **Model Evaluation** | `train_test_split`, `accuracy_score` |
| **Visualization** | Matplotlib |
| **Dataset** | `Crop_recommendation.csv` |

---

## 🔄 How It Works

1. The application loads the `Crop_recommendation.csv` dataset using pandas.
2. It selects seven input features: `N`, `P`, `K`, `temperature`, `humidity`, `ph`, and `rainfall`.
3. The dataset is split into training and testing sets using an 80:20 split.
4. `StandardScaler` scales the feature values.
5. A `RandomForestClassifier` with 100 trees is trained on the training data.
6. The model's accuracy is calculated using the test set and shown in the application.
7. The user adjusts the soil and climate parameters in the Tkinter interface and selects **Get Recommendation**.
8. The model predicts the most suitable crop and displays its confidence score and alternative predictions.
9. Matplotlib charts compare the entered parameters with the average feature values for the recommended crop and display the top five predicted crops.

---

## 📁 Project Structure

```text
Crop-recommendation-System/
├── Crop_recommendation.csv    # Dataset used to train and test the model
├── crop_recommendation.py     # GUI, model training, prediction, and charts
└── README.md                 # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or later recommended
- pip (Python package installer)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Nikhilkumar0101/Crop-recommendation-System.git

# 2. Open the project directory
cd Crop-recommendation-System

# 3. (Optional) Create a virtual environment
python -m venv venv
```

Activate the virtual environment:

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

Install the required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib
```

### Run the Application

```bash
python crop_recommendation.py
```

The desktop application will open. Adjust the soil and climate values using the sliders, then click **Get Recommendation** to view the prediction.

> **Important:** Keep `Crop_recommendation.csv` in the same directory as `crop_recommendation.py`. The application looks for the dataset in that location.

---

## 🧾 Input Parameters

| Parameter | Description |
|---|---|
| **Nitrogen (N)** | Nitrogen content in the soil |
| **Phosphorus (P)** | Phosphorus content in the soil |
| **Potassium (K)** | Potassium content in the soil |
| **Temperature** | Environmental temperature in °C |
| **Humidity** | Relative humidity in % |
| **Soil pH** | Soil acidity or alkalinity |
| **Rainfall** | Rainfall value used by the dataset |

The application uses these seven parameters to estimate which crop label best matches the supplied conditions.

---

## 📊 Model Details

| Setting | Value |
|---|---|
| **Algorithm** | Random Forest Classifier |
| **Number of Trees** | 100 |
| **Test Split** | 20% |
| **Random State** | 42 |
| **Evaluation Metric** | Accuracy |
| **Prediction Output** | Most likely crop plus alternative predictions |

The displayed accuracy is calculated at startup from the test split. Actual performance depends on the dataset and the way the model is evaluated.

---

## 🖥️ Application Output

After a prediction, the interface displays:

- The recommended crop and a crop-specific emoji
- The model's confidence score
- Alternative crop predictions
- A bar chart comparing user-provided values with average values for the recommended crop in the dataset
- A chart showing the top five crop predictions and their confidence scores

---

## ⚠️ Notes and Limitations

- Recommendations depend on the dataset and the input values supplied by the user.
- The confidence score is the model's predicted probability, not a guarantee of crop success.
- This is an educational machine learning project and should not replace local agricultural expertise or soil testing.
- The dataset must be present locally for the application to start.

---

## 🔮 Future Improvements

- Add more detailed crop information and cultivation guidance
- Include fertilizer recommendations based on soil parameters
- Improve validation and model evaluation using cross-validation
- Compare Random Forest with other classification algorithms
- Export prediction results to CSV or PDF
- Add a web-based interface for easier access

---

## 👨‍💻 Author

**Nikhil Kumar**  
MCA (AI/ML) Student

- **GitHub Repository:** https://github.com/Nikhilkumar0101/Crop-recommendation-System
