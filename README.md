# Tourism Experience Analytics - ML

## Project Overview:
This project leverages machine learning and data analytics to enhance user experiences in the tourism industry. It focuses on predicting user satisfaction (regression), classifying visit modes (classification), and recommending personalized tourist attractions (recommendation system) — all integrated into a unified Streamlit application.

**Domain:** **Tourism**

**Goals:**
- Predict user attraction ratings
- Classify the mode of visit (e.g., Business, Family, Friends)
- Recommend personalized tourist attractions

**Key Input Features:**
- **User demographics:** Continent, Country, City
- **Visit details:** Year, Month, Visit Mode
- **Attraction attributes:** Type, Location, Ratings

## Problem Statement:
Tourism agencies and travel platforms aim to enhance user experiences by leveraging data to provide personalized recommendations, predict user satisfaction, and classify potential user behavior. This project involves analyzing user preferences, travel patterns, and attraction features to achieve three primary objectives: regression, classification, and recommendation.

### Business Use Cases:
- **Personalized Recommendations:** Suggest attractions based on users' past visits, preferences, and demographic data, improving user experience.
- **Tourism Analytics:** Provide insights into popular attractions and regions, enabling tourism businesses to adjust their offerings accordingly.
- **Customer Segmentation:** Classify users into segments based on their travel behavior, allowing for targeted promotions.
- **Increasing Customer Retention:** By offering personalized recommendations, businesses can boost customer loyalty and retention.

### Technologies Used
- **Python**: Core language for data processing and model development  
- **Pandas**: Data manipulation and preprocessing (merging, grouping, filtering)  
- **NumPy**: Numerical operations and efficient array handling  
- **Scikit-learn**: Implementation of regression, classification, and evaluation metrics  
- **Scikit-surprise**: For collaborative filtering and recommendation systems  
- **Matplotlib**: Static visualizations for EDA and model interpretation  
- **Seaborn**: Statistical and comparative data visualizations  
- **Plotly**: Interactive and dynamic visualizations used in the dashboard  
- **Openpyxl** : Reading/writing Excel data  
- **Streamlit**: Framework to build an interactive web application/dashboard  
- **VS Code**: Integrated Development Environment (IDE) for development

### Setup Instructions
To manage dependencies separately from the global Python environment, this project requires **Python 3.11** version on the system.

**Create a virtual environment** 
Inside your project folder, run the following command: python -m venv env

**Activate the environment** 
On Windows: - .\env\Scripts\activate

### Installation Instructions
To run the Tourism Experience Analytics project, install the required libraries using pip.
**pip install streamlit pandas plotly openpyxl numpy seaborn matplotlib scikit-learn scikit-surprise**

#### Breakdown of Packages Used
Breakdown of Packages Used:

- **streamlit** → Web app framework (Dashboard/Frontend)
- **pandas** → Data Manipulation and Analysis
- **plotly** → Interactive Visualizations for charts
- **openpyxl** → Read/Write Excel files 
- **numpy** → Numerical operations and Array Handling
- **seaborn** → Statistical plots (Used in EDA)
- **matplotlib** → Core plotting library for static charts
- **scikit-learn** → ML models (Regression, Classification)
- **scikit-surprise** → For building collaborative filtering Recommendation Systems

### Code File Structure
**tourism_data_explorer.ipynb** – Jupyter Notebook for data cleaning, exploratory analysis (EDA), model building (regression, classification, recommendation)
**tourism** – Streamlit frontend folder containing the interactive web application (dashboard, predictions, and recommendations)

### Data Description
The project uses a unified dataset composed of multiple interconnected tables to analyze and predict tourism-related user behavior. Below is a summary of the key datasets:

**1. Transaction Data**
- Purpose: Records user visits and attraction ratings.
- Key Fields: TransactionId, UserId, VisitYear, VisitMonth, VisitMode, AttractionId, Rating
- Usage: Basis for prediction tasks like rating prediction and visit mode classification.

**2. User Data**
- Purpose: Contains user location info.
- Key Fields: UserId, ContinentId, RegionId, CountryId, CityId
- Usage: Supports demographic-based analysis and recommendations.

**3. Item (Attraction) Data**
- Purpose: Information about tourist attractions.
- Key Fields: AttractionId, AttractionCityId, AttractionTypeId, Attraction, AttractionAddress
- Usage: Used for personalized attraction recommendations.

**4. Attraction Type Data**
- Purpose: Categories of attractions.
- Key Fields: AttractionTypeId, AttractionType
- Usage: Enables filtering and classification by type (e.g., Museum, Park).

**5. Visit Mode Data**
- Purpose: Defines modes of visit.
- Key Fields: VisitModeId, VisitMode
- Usage: Used in visit mode classification and user profiling.

**6. City, Country, Region, Continent Data**
- Purpose: Hierarchical location mapping.
- Key Fields: CityId, CityName
    - CountryId, Country, RegionId
    - RegionId, Region, ContinentId
    - ContinentId, Continent
- Usage: Links users and attractions geographically for contextual analysis.

### Approach
**Data Cleaning & Preprocessing**
- Handled missing values and resolved inconsistencies in city names, visit modes, and attraction types.
- Standardized date formats and corrected outliers in ratings.
- Encoded categorical variables (e.g., VisitMode, Country, AttractionType).
- Merged relevant tables to form a consolidated dataset.
- Normalized numerical fields for model training.

**Exploratory Data Analysis (EDA)**
- Visualized user distribution by region and continent.
- Analyzed popular attraction types and rating trends.
- Explored links between demographics and visit modes.

**Model Training**
- Regression: Predicted user attraction ratings.
- Classification: Predicted visit mode using user and visit data.
- Recommendation: Implemented collaborative and content-based filtering to suggest attractions.

**Model Evaluation**
- Regression: Evaluated using R², MSE.
- Classification: Evaluated using Accuracy, Precision, Recall, F1.
- Recommendation: Used RMSE, MAP for performance check.

**Deployment**
Built a Streamlit app to:
- Predict visit mode from user inputs.
- Recommend attractions based on user history/preferences.
- Display dynamic visualizations of trends, locations, and preferences.

### Model Evaluation & Metrics
The models were evaluated using standard performance metrics:
- **Regression Task**: Assessed using R² score and Mean Squared Error (MSE)
- **Classification Task**: Evaluated with Accuracy, Precision, Recall, and F1-score
- **Recommendation System**: Evaluated using RMSE and MAP (Mean Average Precision)
The results provided meaningful insights and were used to enhance the model and user experience in the app.

### Streamlit Features
This interactive web application is built using Streamlit and integrates data analytics, machine learning, and visualization. The app includes the following pages:
- **Home** - Welcome screen with a brief overview of the project’s purpose and goals.
- **Data Explorer**
     - Browse raw datasets such as User, Transaction, Attraction, etc.
     - Filter, clean, and download data.
     - View basic visual summaries.
- **User Summary Statistics**
     - View personalized attraction suggestions for a selected user.
     - Analyze user behavior and segmentation patterns.
- **Analyze Trends**
     - Explore tourism trends by year, region, country, visit mode, etc.
     - Select a year to visualize travel patterns using dynamic charts.
- **Predict Visit Mode**
     - Predict the user's visit reason (e.g., Business, Family, Friends) using a classification model.
     - Displays model predictions along with accuracy metrics.
- **Predict Ratings**
     - Estimate user satisfaction (rating 1–5) for a given attraction using a regression model.
     - Shows predicted values and model evaluation metrics.
- **Get Recommendations**
     - Suggest new tourist attractions for a selected user.
     - Combines collaborative and content-based filtering for personalized results.
 
### Usage: How to Run the Project
To run the project, open the terminal and use the following command: streamlit run app(file name).py
