## Section 1: Machine Learning

#What is Machine Learning?
#Machine Learning (ML) is a technique that allows computers to learn patterns from data and make predictions or decisions 
    #without being explicitly programmed with rules

#Purpose:
#To make the system intelligent enough to give outputs (like recommendations or predictions) based on experience (data), 
    #without manually writing logic for every scenario.

#How It Works:
#Instead of writing fixed instructions, we provide the system with:
# - Input features (Data points)
# - A target or expected output (For Supervised Learning)
#The model then learns the relationship between the inputs and outputs from this historical data.
#Once trained, it can predict the output for new, unseen inputs.

#Why Machine Learning?
#ML is useful when rules are too complex or data-driven patterns change frequently.
#It allows systems to adapt and improve over time by learning from new data.

#In the Context of This Project:
#Machine Learning was used to:
# - Predict user satisfaction in the form of ratings (Regression)
# - Classify the purpose of a user’s visit (Classification)
# - Recommend suitable tourist attractions (Recommendation system)

#Benefits in Our Use Case:
# - Personalized user experience based on data
# - Smarter predictions of user behavior
# - Automated decision-making for large-scale tourism data

#Summary:
#ML replaced manual logic with learned behavior from actual user interaction data, making our tourism dashboard 
   #intelligent, adaptive, and scalable.


## Section 2: Types of Machine Learning Used

#Overview
#Supervised Learning
   #Regression     → Predicting Ratings
   #Classification → Predicting Visit Mode

#Unsupervised Learning
   #Recommendation System → Suggesting Tourist Attractions (via collaborative filtering)

#Supervised Learning
#Supervised learning is used when the dataset contains both input features and the corresponding target variable.
#The model learns the mapping from features to the target using historical labeled data.

#Regression – Predicting Ratings

#What is Regression?
#Regression is a supervised learning approach used when the target is a continuous numeric value.
#The model tries to learn the relationship between input features and the target to make predictions.

#Why Regression in This Project?
#We predict how much a user would rate a tourist attraction (e.g., 1 to 5).
#Since the rating is numeric, regression is the appropriate technique.

# 📊 Model Used: Linear Regression (from sklearn.linear_model)

#What is Linear Regression?
#Linear Regression is a commonly used regression model that tries to find the best-fitting straight line that predicts 
    #the target variable based on the input features.

#How We Used It:
# - Used `LinearRegression()` from scikit-learn.
# - Trained on features: ContinentId, RegionId, CountryId, CityId, VisitYear, VisitMonth, AttractionTypeId, etc.
# - The model learned internal coefficients to map features → rating.

#Preprocessing Steps:
# - Handled missing values via `dropna()` or `fillna()`.
# - Categorical columns like 'AttractionTypeId' were encoded using LabelEncoder.
# - Feature and target columns were selected based on relevance to user behavior and rating.
# - Split data into train/test sets with `train_test_split()`.

#Evaluation Metrics:
# - MSE, RMSE, R² Score

#Evaluation Metrics Used:
# - Mean Squared Error (MSE)        : Measures average squared error between predictions and actual ratings.
# - Root Mean Squared Error (RMSE)  : Square root of MSE; more interpretable as it's in the same scale as ratings.
# - R² Score (R-squared)            : Indicates how well the model explains the variation in the target.


#Libraries Used:
# - scikit-learn : LinearRegression, train_test_split, evaluation metrics
# - pandas       : Data handling and preparation
# - numpy        : Numeric operations (optional)

#Note:
#We did not manually define any regression formula; we used `LinearRegression()` from sklearn, 
   #which internally learns and applies the mathematical model.


# 🎯 Classification – Predicting Visit Mode

#What is Classification?
#Classification is a type of supervised learning where the target variable is **categorical**.
#The model learns to assign each input to a specific class or category.

#Why Classification in This Project?
#In our project, we aimed to predict the **visit mode** (e.g., Business, Family, Friends) for a user based on their
    #location and visit details. Since the target is a **category**, classification is the correct choice.

#📊 Model Used: Random Forest Classifier (from sklearn.ensemble)

#What is Random Forest?
#Random Forest is an ensemble learning method that builds multiple decision trees and combines their outputs.
#It improves accuracy and prevents overfitting compared to a single tree.

#How We Used It:
# - We used `RandomForestClassifier()` from scikit-learn.
# - The model was trained using location-based and temporal features (like ContinentId, CountryId, VisitMonth).
# - It predicts the most likely visit mode for a user.

#Preprocessing Steps Applied:
# - Loaded merged data from `transactions` and `user` sheets.
# - Selected features: `ContinentId`, `RegionId`, `CountryId`, `CityId`, `VisitYear`, `VisitMonth`.
# - Target variable: `VisitMode`.
# - Applied `LabelEncoder()` to convert text-based visit modes to numeric form.
# - Used `train_test_split()` to divide data into 80% training and 20% testing sets.

#Evaluation Metrics Used:
# - Accuracy Score          : Overall percentage of correct predictions.
# - Classification Report   : Includes precision, recall, and F1 score for each class.
# - Confusion Matrix        : Visual representation of correct vs incorrect predictions.

# | Metric               | What It Means                                                   |
# | -------------------- | --------------------------------------------------------------- |
# | **Accuracy**         | % of correct predictions out of total                           |
# | **Precision**        | Of all predicted as "Family", how many actually were "Family"?  |
# | **Recall**           | Of all true "Family" visits, how many did we correctly predict? |
# | **F1 Score**         | Balanced average of precision and recall                        |
# | **Confusion Matrix** | Shows where the model is making wrong predictions               |

# 🔄 Classification Workflow Summary

# 1️⃣ Data Preparation --> Select features and assign VisitMode as target.
# 2️⃣ Encoding         --> Convert VisitMode text labels to numeric via LabelEncoder.
# 3️⃣ Train-Test Split --> Split into 80% training, 20% testing sets.
# 4️⃣ Model Training   --> Train RandomForestClassifier on training data.
# 5️⃣ Prediction       --> Predict VisitMode for test data or new inputs.
# 6️⃣ Evaluation       --> Compute Accuracy, Precision, Recall, F1 Score, and Confusion Matrix.

# 🔄 Classification Workflow Summary (Visit Mode Prediction)

# 📋 Steps Followed:

# 1️⃣ Data Preparation
# This step involves selecting useful features (like ContinentId, RegionId, VisitMonth, etc.)
# and assigning the target variable 'VisitMode'. Only clean and relevant rows are used.

# 2️⃣ Encoding
# Since the VisitMode is a text column (like "Family", "Business"), it must be converted to numbers.
# We used LabelEncoder from sklearn to transform these into integers the model can understand.

# 3️⃣ Train-Test Split
# To evaluate performance fairly, we split the dataset:
# - 80% used for training (learning the pattern)
# - 20% used for testing (checking generalization)
# Done using train_test_split().

# 4️⃣ Model Training
# A RandomForestClassifier was used to learn how input features relate to visit modes.
# The model builds multiple decision trees and combines their outputs for accurate prediction.

# 5️⃣ Prediction
# Once trained, the model was used to predict the visit mode for new user inputs.
# This is integrated into the Streamlit app for live predictions.

# 6️⃣ Evaluation
# Model performance was evaluated using:
# - Accuracy Score
# - Precision, Recall, F1 Score (from classification_report)
# - Confusion Matrix to visualize true vs predicted values

# These metrics help us understand if the model is performing well across all visit mode categories.


# 📦 Tools Used in Project

# | Tool                   | Purpose                                      |
# |------------------------|----------------------------------------------|
# | RandomForestClassifier | Model for predicting visit mode             |
# | LabelEncoder           | Convert text labels (VisitMode) to numbers   |
# | train_test_split       | Split data into training and testing sets    |
# | accuracy_score,        | Evaluate model performance (accuracy,        |
# | classification_report  | precision, recall, F1 score)                 |
# | ConfusionMatrixDisplay | Visualize true vs. predicted class counts    |


# 📊 Visualization:
# - Displayed bar chart of visit mode distribution using Plotly.
# - Used confusion matrix for evaluation insight.

#Output:
# The model predicts the most likely visit mode (e.g., "Family", "Business") for new user visit records,
# helping understand travel behavior.

#Libraries Used:
# - scikit-learn : RandomForestClassifier, train_test_split, LabelEncoder, accuracy_score, classification_report
# - pandas       : Data loading, preprocessing
# - matplotlib / seaborn / plotly : Visualization

#Note:
#Classification was implemented and evaluated using **random forest**, and label encoding was applied to handle 
   #categorical targets. The predictions were integrated into the Streamlit app for live inference.



#Unsupervised Learning
#Unsupervised learning is used when the dataset contains only input features and no predefined target variable.
#The model tries to discover hidden patterns or groupings within the data without being told what to look for.

#🎯 Recommendation System – Suggesting Tourist Attractions

#What is a Recommendation System?
#A recommendation system analyzes user behavior and preferences to suggest relevant items.
#It does not rely on labeled target variables, making it a type of unsupervised learning.

#Why a Recommendation System in This Project?
#The goal was to recommend tourist places to users based on their past preferences or similar user behavior.
#This improves user engagement and delivers personalized suggestions in the tourism domain.

#Short Note: To recommend attractions based on past user ratings and similar user behavior.

# 📊 Techniques Used:
# - **Collaborative Filtering** using the `Surprise` library (user-based recommendations)
# - **Cosine Similarity** to find similar users or items
# - **Top-N Recommendation** logic to return the most relevant places for each user

#How It Works:
# - Collected ratings data (`UserId`, `PlaceId`, `Rating`)
# - Used the `Surprise` library to build a collaborative filtering model (e.g., `KNNBasic`)
# - Calculated similarities between users or attractions
# - Generated a ranked list of top tourist places for each user

#Preprocessing and Data Used:
# - Ratings were extracted from the ratings sheet or interaction data.
# - User-item matrix was created where rows = users, columns = places, values = ratings.
# - Similarity scores were computed using cosine distance or Surprise’s internal algorithms.

#Output:
# The system predicts which attractions the user is likely to enjoy, even if they haven’t visited them yet.
# These recommendations were displayed in the Streamlit dashboard with place names and predicted ratings.

#📦 Libraries Used:
# - `surprise`: Dataset, Reader, KNNBasic, prediction logic
# - `sklearn.metrics.pairwise`: cosine_similarity (for manual similarity logic)
# - `pandas`, `numpy` for matrix handling
# - `Streamlit` to display Top-N recommendations in the UI

#Note:
# This approach does not need predefined target labels, making it an example of unsupervised learning.
# It learns patterns based on existing ratings and behavior, not manual labeling.

#Section Summary: ML Approaches Used

# | Task                     | ML Type                     | Model Used                 | Output                        |             
# | Predict user rating      | Supervised - Regression     | LinearRegression           | Continuous rating (e.g., 4.5) |
# | Predict visit mode       | Supervised - Classification | RandomForestClassifier     | Visit category (e.g., Family) |
# | Recommend attractions    | Unsupervised                |  Surprise + Cosine Similarity| Top-N tourist suggestions   |



## Section 3: Core Machine Learning Concepts Used in the Project

#Feature:
#A feature is an input variable used by the machine learning model to make predictions.
#In this project, features include:
# - Location-based: ContinentId, RegionId, CountryId, CityId
# - Time-based: VisitMonth, VisitYear
# - Attraction-based: AttractionTypeId, VisitDuration

#Target:
# The target is the output variable the model is trying to predict.
# - For Regression     → Target: Rating (a continuous number)
# - For Classification → Target: VisitMode (a category like Family, Business)

#Train-Test Split

#Why We Split the Data:
#To evaluate whether the model generalizes well to unseen data.
#It is the process of dividing the dataset into two parts:
# - Training Set: Teaches the model (usually 70–80% of the data)
# - Test Set    : Checks model performance on new data (20–30%)

#Why It Matters:
# - Helps check if the model generalizes well beyond the data it learned from

#Tool Used:
# - train_test_split() from sklearn.model_selection
# - Used with `random_state=42` to ensure reproducibility

#Data Preprocessing

#Missing Value Handling (General Info)
#Missing values occur when data is not recorded or is unavailable.
# Handling them properly is important to avoid errors and ensure model accuracy.

#Methods to Handle Missing Data:
# - dropna(): Removes rows or columns with missing values
# - fillna(): Fills missing values with a constant (e.g., 0), mean, median, or mode

#Why It Matters:
# - Unhandled missing data can lead to biased results or model failure

#Encoding:
# Encoding converts non-numeric categorical variables into numeric form so that machine learning models can process them.

#Encoding Categorical Variables:
# - Used LabelEncoder() from sklearn.preprocessing
# - Applied to columns like VisitMode, AttractionTypeId for model compatibility

#Feature Selection: (General Info)
#Feature selection is the process of identifying the most relevant input variables (features) for training a ML model.

#Why It Matters:
# - Improves model accuracy by removing noise
# - Reduces overfitting and training time
# - Makes the model simpler and easier to interpret

#How Features Are Selected:
# - Based on domain knowledge or business context
# - Using statistical methods (e.g., correlation with the target)
# - Avoiding redundant, irrelevant, or highly correlated features

#Feature Selection: (In this Project)
# - Selected only relevant and useful columns based on:
#   - Domain knowledge
#   - Correlation with target
#   - Avoiding high-dimensional or duplicate columns

#What is Correlation?
#Correlation is a statistical measure of the relationship between two variables.
# - High correlation with target      → Good predictor
# - High correlation between features → Avoid (redundant)

# 📈 In Feature Selection:
# - We look for features that have high correlation with the target (helpful predictors)
# - And low correlation with each other (to avoid redundancy or multicollinearity)

#Example:
# If VisitMonth is highly correlated with VisitMode, it may be a useful feature to include.


#Hyperparameters

#What are Hyperparameters?
# - Hyperparameters are configuration values set **before training**.
# - They control how the model learns, unlike model parameters which are learned from data.

#Why They Matter:
#Tuning them can improve accuracy and reduce overfitting.

#Hyperparameters Used in This Project:

#For RandomForestClassifier:
# - n_estimators=100 → Number of decision trees
# - max_depth=None   → Trees grow fully unless stopped by other limits
# - criterion='gini' → Measures quality of splits

#For LinearRegression:
# - No major hyperparameters manually set — Used sklearn's default config

#For Surprise Recommender (KNNBasic):
# - k → Number of nearest neighbors (default or tuned)
# - similarity_options: Defines the similarity metric (e.g., cosine)


#Model Evaluation Metrics

#📈 Regression Metrics:
# - MSE (Mean Squared Error)      : Measures average squared prediction error
# - RMSE (Root Mean Squared Error): Easier to interpret since it's in the same unit as rating
# - R² Score                      : Represents how well the model explains variation in the target

#📊 Classification Metrics:

# 📊 General Definitions of Evaluation Metrics

# - Accuracy:
#   The percentage of correct predictions made by the model out of all predictions.
#   Formula: (TP + TN) / (TP + TN + FP + FN)

# - Precision:
#   Out of all instances the model predicted as positive, how many were actually positive.
#   Formula: TP / (TP + FP)
#   Helps reduce false positives.

# - Recall:
#   Out of all actual positive instances, how many did the model correctly identify.
#   Formula: TP / (TP + FN)
#   Helps reduce false negatives.

# - F1 Score:
#   A balanced average of Precision and Recall.
#   Useful when you need to balance both false positives and false negatives.
#   Formula: 2 * (Precision * Recall) / (Precision + Recall)

# - Confusion Matrix:
#   A 2D table showing the number of correct and incorrect predictions
#   broken down by each class (actual vs predicted).
#   Helps visualize where the model is making mistakes.

#📊 Classification Metrics: (In this project)
# - Accuracy        : Overall % of correct predictions
# - Precision       : How many predicted "positives" were truly correct
# - Recall          : How many actual "positives" were correctly predicted
# - F1 Score        : Harmonic mean of precision and recall (balances both)
# - Confusion Matrix: Table showing predicted vs actual class distribution

#Tools Used for Evaluation:
# - sklearn.metrics: accuracy_score, precision_score, recall_score, f1_score, classification_report
# - ConfusionMatrixDisplay, plot_confusion_matrix, mean_squared_error, r2_score
# - Plotly and seaborn used for visualizing trends, distributions, and matrix heatmaps


#Summary:
#This section covered core ML concepts as applied in this project:
# - Feature-target setup for different models
# - Data splitting and encoding
# - Handling missing values and choosing features
# - Using hyperparameters to control model behavior
# - Evaluating models with proper metrics for both regression and classification

#These concepts ensured our ML pipeline was structured, efficient, and reliable in real-time predictions 
    #within the Streamlit dashboard.


## Section 4: Model Logic Flow – End-to-End Steps

#This section explains how each model works internally, from receiving user input to generating predictions 
  #or recommendations.


# 🎯 1. Regression Model Logic Flow (Predicting Ratings)

# Step 1: Input
# - User provides inputs like AttractionTypeId, TravelTime, VisitDuration, etc.

# Step 2: Preprocessing
# - Missing values (if any) are handled
# - Categorical variables (like AttractionTypeId) are label encoded
# - Features are organized into a proper DataFrame

# Step 3: Model Prediction
# - Preprocessed data is passed into the trained LinearRegression model
# - Model uses the learned weights to predict a numeric rating

# Step 4: Output
# - The predicted rating (e.g., 4.2) is displayed in the Streamlit dashboard


# 🎯 2. Classification Model Logic Flow (Predicting Visit Mode)

# Step 1: Input
# - User provides details like Region, Country, City, VisitMonth, VisitYear

# Step 2: Preprocessing
# - Missing values are handled
# - VisitMode column is encoded using LabelEncoder
# - Input features are structured in the same order as training

# Step 3: Model Prediction
# - Data is passed into the trained RandomForestClassifier
# - The model outputs the most likely VisitMode (e.g., "Family")

# Step 4: Output
# - The predicted class label is decoded (if needed) and shown to the user


# 🎯 3. Recommendation System Logic Flow (Suggesting Tourist Places)

# Step 1: Input
# - User ID is taken as input or inferred from rating history

# Step 2: Data Preparation
# - Ratings data is organized as a user-item matrix (UserId, PlaceId, Rating)
# - Surprise library prepares the dataset using Reader + Dataset.load_from_df()

# Step 3: Similarity Calculation
# - Using Surprise’s KNNBasic algorithm or cosine_similarity
# - Finds users/items similar to the active user

# Step 4: Top-N Recommendation Generation
# - Predicts ratings for unrated places
# - Sorts and selects top N highest-rated suggestions

# Step 5: Output
# - List of top-N recommended tourist attractions is shown in Streamlit


#Summary:
# Each model in the system follows a clear pattern:
# - Receive input → Preprocess → Predict using trained model → Display result
# This modular logic ensures reusability, readability, and real-time user interaction.


## Section 5: Streamlit Integration – ML + UI Connection

# 🖥️ What is Streamlit?
# Streamlit is an open-source Python framework used to build interactive web applications
# for machine learning and data analysis with minimal code.

#Why We Used It:
# - Easy to integrate ML models with user input forms
# - Live, real-time predictions
# - Visualizations (charts, tables, maps) with interactive components

# ---------------------------------------------------------

#How Streamlit Works in This Project:

# 1️⃣ Input Collection
# - Users select or enter values using Streamlit widgets:
#   - st.selectbox() for dropdowns (e.g., VisitMode, Region)
#   - st.slider(), st.number_input(), st.text_input() for numeric/text data
#   - st.button() or st.form_submit_button() to trigger predictions

# 2️⃣ Preprocessing Inside the App
# - The inputs are structured as a DataFrame (just like training data)
# - Label encoding or transformation is applied if required (e.g., VisitMode)
# - Missing values are handled where applicable

# 3️⃣ ML Model Prediction
# - The model (already trained and loaded) receives the input features
# - Prediction is done using `.predict()` for classification/regression models
# - For recommendation, Surprise or cosine similarity logic returns Top-N places

# 4️⃣ Displaying Results
# - Predictions are shown using:
#   - st.success(), st.write(), st.markdown() for text
#   - st.dataframe(), st.table() for tabular results
#   - plotly or matplotlib visualizations for charts


# 📊 Streamlit Page Structure (Example):

# - 🏠 Home: Introduction and project overview
# - 🔍 Data Explorer: View raw data, filters, and insights
# - 📈 Predict Ratings: Regression model interface (Linear Regression)
# - 🧮 Predict Visit Mode: Classification model (Random Forest)
# - 🌍 Get Recommendations: Top-N places using collaborative filtering
# - 📊 User Statistics: Charts, trends, and summaries


#Example: Visit Mode Prediction Page (Classification)

# - User selects RegionId, CountryId, VisitMonth, etc.
# - Clicks "Predict" button
# - Streamlit preprocesses inputs and passes them to RandomForestClassifier
# - The predicted VisitMode (e.g., "Business") is displayed in real time


#Summary:
# Streamlit bridges the gap between ML models and end users.
# It collects input, processes it, runs prediction logic, and returns output — all within a clean UI.
# This makes the entire ML system interactive, user-friendly, and production-ready.
