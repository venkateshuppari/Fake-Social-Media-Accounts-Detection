# Fake Social Media Accounts Detection

## Overview

This project implements a machine learning solution to detect and classify fake social media accounts. It leverages multiple classification algorithms and provides an interactive web-based interface for real-time predictions.

**Definition**: A fake social media account is an inauthentic profile created for deception, spam, or manipulation, as opposed to genuine user accounts representing real individuals or legitimate organizations.

## Key Components

1. **Pre-trained ML Models** — Four classification algorithms trained on labeled account data
2. **Streamlit Web Application** — Interactive user interface for model selection and predictions
3. **Jupyter Notebook** — Complete pipeline for exploratory data analysis, feature engineering, and model training
4. **Dataset** — Example fake and real user account data with extracted features

## Methodology & Workflow

### Data Pipeline
- **Data Preparation**: Raw account features are merged, cleaned, and preprocessed from CSV files
- **Feature Engineering**: Extraction and transformation of meaningful attributes from account metadata (e.g., profile characteristics, engagement patterns, account age)
- **Preprocessing**: Numerical scaling, categorical encoding, and handling missing values to prepare data for model input

### Machine Learning Models

Each model is trained using supervised learning with labeled examples (fake vs. real accounts):

- **Decision Tree** — A rule-based model that recursively splits data on features to make decisions; interpretable but prone to overfitting
- **Random Forest** — An ensemble of decision trees that reduces overfitting and improves robustness through majority voting
- **Logistic Regression** — A probabilistic linear classifier that outputs prediction confidence scores between 0 and 1
- **Support Vector Machine (SVM)** — A non-linear classifier that finds optimal decision boundaries in high-dimensional feature space

### Model Persistence

Models are serialized using **joblib**, a Python library for efficient saving/loading of sklearn objects. This allows trained models to be loaded and used for predictions without retraining.

## Repository Structure

```
.
├── streamlit_app.py                          # Interactive web application
├── Fake_social_media_acount_detection.ipynb  # Data exploration & model training notebook
├── decision_tree.joblib                      # Trained Decision Tree model
├── random_forest.joblib                      # Trained Random Forest model
├── logistic_regression.joblib                # Trained Logistic Regression model
├── svm_model.joblib                          # Trained SVM model
├── prediction_history.csv                    # Log of app predictions (auto-generated)
├── requirements.txt                          # Python dependencies
├── README.md                                 # This file
└── dataset/
    ├── fake_users.csv                        # Labeled fake account records
    └── real_users.csv                        # Labeled real account records
```

## Requirements

- **Python**: 3.8 or higher
- **Dependencies**: Listed in `requirements.txt`

Install all required packages:

```bash
pip install -r requirements.txt
```

## Getting Started

### Option 1: Interactive Web Application

1. Install dependencies (see above)
2. Navigate to the project directory
3. Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

4. Open your browser to the URL displayed (typically `http://localhost:8501`)
5. Select a model from the sidebar and enter account features to classify accounts as fake or real

### Option 2: Programmatic Use

Load and use a trained model directly in Python:

```python
import joblib
import pandas as pd

# Load a pre-trained model
model = joblib.load('random_forest.joblib')

# Prepare input data (must match training features and preprocessing)
# Example: X = pd.DataFrame({'feature1': [value], 'feature2': [value], ...})
# Ensure features are in the same order and format as used during training

# Make prediction
prediction = model.predict(X)  # Returns 0 (real) or 1 (fake)
probability = model.predict_proba(X)  # Returns confidence scores
```

**Important**: Input features must be preprocessed identically to the training data (scaling, encoding, feature order).

## Dataset Details

The `dataset/` folder contains two CSV files representing the training data:

- **fake_users.csv**: Account records labeled as inauthentic (fake = 1)
- **real_users.csv**: Account records labeled as authentic (real = 0)

**Usage**: These files serve as examples. To retrain models:
1. Merge both files
2. Apply feature engineering and preprocessing
3. Train models using sklearn
4. Export trained models with joblib

## Model Training & Retraining

The Jupyter notebook (`Fake_social_media_acount_detection.ipynb`) contains the complete training pipeline:

1. **Load & Explore**: Import and analyze fake/real user datasets
2. **Preprocessing**: Handle missing values, encode categorical variables, scale numerical features
3. **Feature Engineering**: Create or select relevant features for classification
4. **Model Training**: Train all four algorithms on the prepared dataset
5. **Evaluation**: Assess accuracy, precision, recall, and F1-score on test data
6. **Model Export**: Save the best-performing models using joblib

To retrain or experiment with new models:
- Open the notebook in Jupyter: `jupyter notebook Fake_social_media_acount_detection.ipynb`
- Modify preprocessing or feature engineering steps
- Run training cells and export updated models

## How the Application Works

1. **User Input**: User selects a model and provides account features via the Streamlit web interface
2. **Feature Preprocessing**: Input data is transformed (scaled, encoded) to match training data format
3. **Prediction**: The selected model classifies the account as fake or real
4. **Result Display**: Prediction and confidence score are shown to the user
5. **Logging** (optional): Prediction, input features, and timestamp are appended to `prediction_history.csv`

## Fake vs. Real Accounts: Key Differences in Society

### Real (Authentic) Accounts
**Characteristics**:
- Created by genuine individuals or legitimate organizations
- Consistent profile information (photo, bio, posting history)
- Natural engagement patterns with meaningful interactions
- Longer account history with gradual follower growth
- Original, authentic content reflecting real opinions and experiences
- Verified information (email, phone, identity verification where applicable)
- Balanced posting frequency and realistic interaction metrics

**Impact on Society**:
- Build genuine communities and authentic discourse
- Foster trust and transparency in social networks
- Enable real-world connections and meaningful relationships
- Support legitimate businesses and creators
- Contribute to informed public opinion and democratic participation

### Fake (Inauthentic) Accounts
**Characteristics**:
- Created with deceptive intent (often automated or purchased in bulk)
- Generic or stolen profile information and photos
- Unnatural engagement patterns (sudden spikes, bot-like behavior)
- Rapid follower accumulation or engagement without reciprocation
- Copy-pasted or AI-generated content with little originality
- Minimal or fabricated personal details
- Engagement primarily through automated interactions (likes, follows, comments)

**Impact on Society**:
- **Misinformation Spread**: Fake accounts amplify false narratives and conspiracy theories
- **Election Interference**: Coordinate inauthentic campaigns to influence political outcomes
- **Market Manipulation**: Artificially inflate engagement metrics for financial gain
- **Cyberbullying & Harassment**: Enable anonymous, consequence-free abuse
- **Fraud & Scams**: Used for phishing, romance scams, credential theft
- **Erosion of Trust**: Undermine confidence in social media platforms and online information
- **Mental Health Impact**: Contribute to social pressure and unrealistic expectations
- **Loss of Authentic Voice**: Drown out genuine user content with spam and propaganda

### Why Detection Matters
Detecting and removing fake accounts is critical for:
- **Platform Integrity**: Ensuring authentic user experience and trust
- **Public Health**: Combating health misinformation and dangerous trends
- **Economic Protection**: Preventing fraud and maintaining fair market competition
- **Democratic Integrity**: Protecting elections from manipulation
- **User Safety**: Reducing harassment, scams, and exploitation

## Common Workflow: Training New Models

```bash
# 1. Edit the notebook with new data or parameters
jupyter notebook Fake_social_media_acount_detection.ipynb

# 2. Run all cells to train models
# 3. Export new models (handled in notebook)

# 4. Test with the web app
streamlit run streamlit_app.py
```

## Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork or Branch**: Create a feature branch for your changes
2. **Testing**: Include unit tests for preprocessing and prediction logic
3. **Documentation**: Update comments and docstrings for clarity
4. **Data Privacy**: Remove or anonymize any sensitive information before committing
5. **Pull Request**: Submit a PR with a clear description of changes and motivation


## Conclusion

The proliferation of fake social media accounts represents a critical challenge to the integrity of digital communities. This project provides a practical machine learning solution to combat this issue by leveraging multiple classification algorithms to identify inauthentic accounts with high accuracy.

### Key Takeaways

1. **Problem Significance**: Fake accounts cause real-world harm through misinformation, fraud, election interference, and erosion of public trust in digital platforms.

2. **Technical Approach**: By combining Decision Trees, Random Forests, Logistic Regression, and SVM models, this project demonstrates that fake accounts can be reliably detected through behavioral and profile analysis.

3. **Practical Applicability**: The Streamlit web interface makes this technology accessible to both technical and non-technical users, enabling widespread adoption for account verification and platform moderation.

4. **Future Improvements**: 
   - Integrate with real-time social media APIs for continuous monitoring
   - Incorporate deep learning models (LSTM, transformers) for advanced pattern detection
   - Extend to multi-platform detection across multiple social networks
   - Implement feedback loops to adapt models as fake account tactics evolve

5. **Responsible Use**: This tool should be used ethically to protect users and platforms, not to unfairly target legitimate accounts or infringe on privacy.

### Call to Action

If you are a platform developer, security researcher, or concerned citizen:
- Use this project as a foundation for your own detection systems
- Contribute improvements and feedback
- Help raise awareness about the dangers of inauthentic online activity
- Support policies and technologies that promote digital authenticity and trust

Together, we can build safer, more trustworthy online communities.

---

**Last Updated**: December 2025
**Version**: 1.0
