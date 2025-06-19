#Import Required Libraries
import streamlit as st                     #For building the web app interface
import pandas as pd                        #For data manipulation
import numpy as np                         #For numerical operations
from datetime import datetime              #For dynamic greeting based on time
import plotly.express as px                #For interactive visualizations
import io                                  #For in-memory file operations
import plotly.io as pio                    #For saving plots as images

#Commands Used
#import streamlit as st        : To create UI elements in the Streamlit app
#import pandas as pd           : For data loading and manipulation
#import numpy as np            : For numerical operations
#from datetime import datetime : To fetch current hour for greeting
#import plotly.express as px   : To build interactive charts and visuals
#import io                     : For in-memory file operations like creating byte streams (used for downloading files or images)
#import plotly.io as pio       : To save Plotly figures as images (e.g., PNG) in-memory or to files

#Short Description: This block loads all required libraries to build the Streamlit app.

#Load Excel Data Function
#Cache the function output to improve app speed and avoid reloading Excel each time
#Load Excel Data Function
#Cache the function output to improve app speed and avoid reloading Excel each time
@st.cache_data  
# 📥 Define function to load data from the Excel file
def load_data():

    # File path to the Excel dataset (use double backslashes in Windows)
    file_path = r"C:\\Users\\Bala Sowntharya\\Downloads\\Tourism_Experience_Analytics_Dataset.xlsx"

    # 📄 Load 'Transaction' sheet into DataFrame
    df_transactions = pd.read_excel(file_path, sheet_name='Transaction')     # Load transaction data

    # 👤 Load 'User' sheet (contains demographic and location info of users)
    df_user = pd.read_excel(file_path, sheet_name='User')                    # Load user data

    # 🏙️ Load 'Cities' sheet (CityId and CityName mapping)
    df_city = pd.read_excel(file_path, sheet_name='Cities')                  # Load city data

    # 🌍 Load 'Countries' sheet (Country info with mapping to region)
    df_country = pd.read_excel(file_path, sheet_name='Countries')            # Load country data

    # 🗺️ Load 'Region' sheet (Region and Continent mapping)
    df_region = pd.read_excel(file_path, sheet_name='Region')                # Load region data

    # 🌐 Load 'Continent' sheet (basic continent reference)
    df_continent = pd.read_excel(file_path, sheet_name='Continent')          # Load continent data

    # 🎯 Load 'Type' sheet (AttractionTypeId and AttractionType)
    df_types = pd.read_excel(file_path, sheet_name='Type')                   # Load attraction type data

    # 🧭 Load 'Mode' sheet (VisitModeId and VisitMode)
    df_mode = pd.read_excel(file_path, sheet_name='Mode')                    # Load visit mode data

    # 🗃️ Load 'Updated_Item' sheet (detailed attraction info)
    df_updated_item = pd.read_excel(file_path, sheet_name='Updated_Item')    # Load additional item/attraction details

    # 🏞️ Load 'Item' sheet (attraction info backup or original structure)
    df_item = pd.read_excel(file_path, sheet_name='Item')                    # Load item data

    # 📦 Return all loaded DataFrames as a dictionary
    return {
        'transactions': df_transactions,    # Key: transactions data
        'user': df_user,                    # Key: user profile/location data
        'city': df_city,                    # Key: city-level data
        'country': df_country,              # Key: country-level data
        'region': df_region,                # Key: region-level data
        'continent': df_continent,          # Key: continent-level data
        'types': df_types,                  # Key: attraction types
        'mode': df_mode,                    # Key: visit mode info
        'updated_item': df_updated_item,    # Key: updated attractions
        'item': df_item                     # Key: original items
    }

#📥 Load and cache all data
data = load_data()


#📂 Unpack all DataFrames from the loaded data dictionary
df_tx = data['transactions']         # Transactions
df_user = data['user']               # User info
df_city = data['city']               # City details
df_country = data['country']         # Country details
df_region = data['region']           # Region info
df_continent = data['continent']     # Continent info
df_types = data['types']             # Attraction types
df_mode = data['mode']               # Visit modes
df_updated_item = data['updated_item']  # Updated attractions
df_item = data['item']               # Original item structure

#Commands Used
# @st.cache_data                    : Caches the function output to improve speed
# pd.read_excel(..., sheet_name=...): Loads specific sheet from Excel as DataFrame

#Key Features Used
# @st.cache_data          : Caches the result of the function to boost performance by avoiding repeated reading of the file
# pd.read_excel()         : Loads individual sheets from Excel into separate pandas DataFrames
# Dictionary return format: Makes it easier to access each DataFrame by name (e.g., data['transactions'])

#Short Description: Loads and returns all required sheets from the Excel file using a cached function.

#Purpose:
#To efficiently load all required data sheets from the Tourism Analytics Excel file, cache them for faster access, and return them in a structured format to be reused across the app

#Sidebar Navigation
st.sidebar.title("Navigation")

#List of app pages
pages = [
    "🏠 Home",
    "🔍 Data Explorer",
    "📊 User Summary Statistics",
    "📈 Analyze Trends",
    "🧮 Predict Visit Mode",
    "📈 Predict Ratings",
    "🌍 Get Recommendations"
    
] #Sidebar multi-page control

#Radio button for page selection
page = st.sidebar.radio("Go to", pages, index=0)

#Commands Used
#st.sidebar.title(...)   : Displays a title on the sidebar
#st.sidebar.radio(...)   : Provides navigation options as radio buttons

#Short Description: Enables user to switch between multiple pages via the sidebar

#🏠Home Page
#Check if current selected page is 'Home'
if page == "🏠 Home":

    #🕒 Get the current hour (0-23) to determine greeting
    current_hour = datetime.now().hour   #Get system time's current hour

    #🌞 Display morning greeting if before 12 PM
    if current_hour < 12:
        st.write("🌞 Good Morning!")     #Show morning greeting

    #☀️ Display afternoon greeting if between 12 PM and 4 PM
    elif 12 <= current_hour < 16:
        st.write("☀️ Good Afternoon!")   #Show afternoon greeting

    #🌙 Display evening greeting for hours 4 PM onwards
    else:
        st.write("🌙 Good Evening!")     #Show evening greeting

    #Key Features Used
    #datetime.now().hour: Gets current hour dynamically to personalize user interaction
    #if-elif-else       : Conditional logic to control which greeting is displayed
    #st.write()         : Streamlit method to display text on the app

    #Purpose:
    #To provide a dynamic greeting message to the user based on the current time of day, making the app feel personalized and friendly

    #Title
    st.title("🌍 Explore Travel Experience!")

    #Introduction Text
    st.markdown("""
    Welcome to the **Tourism Experience Analytics Dashboard** 🧳

    This tool allows you to:
    - Analyze tourism trends by region, city, and attraction types
    - Predict your **likely visit mode** (e.g., Business, Family, Solo)
    - Get **personalized attraction recommendations**
    """)

    #Expandable section for instructions
    #📖 Create an expandable section titled "How to Use This App"
    with st.expander("📖 How to Use This App"):   #Creates a collapsible section to display guidance or help content

    #📝 Display updated instructions for navigating the app using markdown formatting
    #Multi-line markdown to show usage instructions with icons and bold text
      st.markdown("""
      Use the **sidebar** to explore the following pages:

       - 🏠 **Home**                  : Landing page with dynamic greetings and project overview  
       - 🔍 **Data Explorer**         : Explore and visualize raw tourism data using filters and charts 
       - 📊 **User Summary Statistics**: Personalized attraction suggestions, user segments, and tourism behavior insights           
       - 📈 **Analyze Trends**        : Estimate user ratings using input-based prediction              
       - 🧮 **Predict Visit Mode**    : Predict a user's travel purpose (e.g., Family, Business)  
       - 📈 **Predict Ratings**       : Predict user ratings based on travel patterns, destination, and attraction type
       - 🌍 **Get Recommendations**   : Explore region-wise travel trends  
    
       """) #End of markdown instructions
      

#🗺️ Streamlit Page Navigation
#Use the sidebar to explore the following pages:
# - 🏠 Home                   : Landing page with greeting message and project overview
# - 🔍 Data Explorer          : Interactive charts and filters to explore raw tourism data
# - 📈 Predict Ratings        : Regression-based model to predict user ratings for attractions
# - 🧮 Predict Visit Mode     : Classification model (Random Forest) to predict travel purpose (e.g., Business, Family)
# - 🌍 Get Recommendations    : Region-wise attraction analysis and popular trends
# - 📊 User Summary Statistics: Collaborative + content-based recommendations, user segments, and behavior insights
  
     
    #📝 Optional Input Sections (Commented for future use)
    # st.subheader("🎫 Let's Personalize Your Experience")
    
    # continents = data['continent']['Continent'].unique()
    # selected_continent = st.selectbox("🌐 Select Your Continent:", options=continents)
    
    # continent_id = data['continent'].loc[
    #     data['continent']['Continent'] == selected_continent, 'ContinentId'].values[0]
    
    # countries_filtered = data['country'][data['country']['RegionId'].isin(
    #     data['region'][data['region']['ContinentId'] == continent_id]['RegionId'])]
    
    # country_list = countries_filtered['Country'].unique()
    # selected_country = st.selectbox("🏳️ Select Your Country:", options=country_list)
    
    # visit_modes = data['mode']['VisitMode'].unique()
    # selected_mode = st.selectbox("🧭 Preferred Visit Mode (optional):", options=np.append(["Not Sure"], visit_modes))
    
    # st.markdown("Inputs will be used for prediction and recommendations on the next pages.")

    #User Input Area
    #st.subheader("🎫 Let's Personalize Your Experience")

    #Continent dropdown
    #continents = data['continent']['Continent'].unique()
    #selected_continent = st.selectbox("🌐 Select Your Continent:", options=continents)

    #Get matching ContinentId
    #continent_id = data['continent'].loc[data['continent']['Continent'] == selected_continent, 'ContinentId'].values[0]

    #Filter countries based on selected continent
    #countries_filtered = data['country'][data['country']['RegionId'].isin(
    #    data['region'][data['region']['ContinentId'] == continent_id]['RegionId']
    #)]

    #country_list = countries_filtered['Country'].unique()
    #selected_country = st.selectbox("🏳️ Select Your Country:", options=country_list)

    #Visit Mode Selection
    #visit_modes = data['mode']['VisitMode'].unique()
    #selected_mode = st.selectbox("🧭 Preferred Visit Mode (optional):", options=np.append(["Not Sure"], visit_modes))

    #Summary Note
    #st.markdown("📌 Your inputs will be used for prediction and recommendations on the next pages.")

    #Commands Used:
    #st.write(...)         : To display dynamic greetings
    #st.title(...)         : To show main page title
    #st.markdown(...)      : For formatting blocks of text
    #st.selectbox(...)     : For dropdown user input
    #with st.expander(...) : To collapse/expand help section

    #Short Description:
    #This Home Page serves as the landing screen of the Tourism Analytics App
    #It loads all required datasets, offers a personalized greeting based on the current time, 
    # displays an app introduction, and provides a clear usage guide via an expandable section
    #Users can navigate through the app via the sidebar and (optionally) input their preferences
        #(continent, country, visit mode) to be used in prediction and recommendation modules on other pages


#🔍 DATA EXPLORER PAGE
#🔄 Check if the selected page is 'Data Explorer' from the sidebar
elif page == "🔍 Data Explorer":  #If user selects the Data Explorer page

    #📌 Display the main title of this section
    st.title("🔍 Data Explorer")  #Page title for UI

    #📝 Provide short intro and instructions to user
    st.markdown("""  
        This section allows you to **explore the raw tourism dataset** interactively.

        You can:
        - 📂 View data from different tables (User, Transaction, Country, etc.)
        - 🔎 Scroll, sort, and search through data
        - 📥 Download the selected dataset as CSV for offline use
        - 📈 Visualize distribution of numeric columns via histogram
    """)  #Markdown text describing the features

    #📦 Define a dictionary of dataset names and corresponding DataFrames from loaded data
    dataset_options = {
        "Transactions": data['transactions'],      #Transactions table
        "User Info": data['user'],                 #User demographic and geo info
        "City Info": data['city'],                 #City ID and Name mapping
        "Country Info": data['country'],           #Country details
        "Region Info": data['region'],             #Region data
        "Continent Info": data['continent'],       #Continent mapping
        "Attraction Types": data['types'],         #Type of attractions
        "Visit Modes": data['mode'],               #Mode of travel
        "Updated Items": data['updated_item'],     #Attractions (updated table)
        "Original Items": data['item']             #Attractions (original structure)
    }  #Dictionary maps UI labels to actual DataFrames

    #📂 User dropdown to choose which dataset to display
    selected_dataset_name = st.selectbox("📁 Select a dataset to explore", options=list(dataset_options.keys()))  #Dropdown for dataset selection

    #📄 Extract the selected DataFrame based on user choice
    selected_df = dataset_options[selected_dataset_name]  #Get corresponding DataFrame

    #🧹 Drop duplicate rows (default step)
    selected_df = selected_df.drop_duplicates()  #Removes duplicate rows

    #🧼 Fill missing text/object values with "Missing"
    selected_df = selected_df.apply(lambda col: col.fillna("Missing") if col.dtype == "object" else col)  #Clean nulls

    #🚨 Handle empty dataset case
    if selected_df.empty:
        st.warning("⚠️ The selected dataset is empty!")  #Show warning if no rows
    else:
        #📊 Display the selected DataFrame as a table in the app
        st.dataframe(selected_df, use_container_width=True)  #Show data interactively

        #📥 Button to download the displayed data as CSV (with error handling)
        try:
            csv = selected_df.to_csv(index=False).encode('utf-8')  #Convert to CSV
            st.download_button(
                label="⬇️ Download This Dataset as CSV",   #Button label
                data=csv,                                   #CSV content
                file_name=f"{selected_dataset_name}.csv",   #Output file name
                mime='text/csv'                             #File type
            )  # Show download button
        except Exception as e:
            st.error(f"❌ Failed to generate CSV: {e}")  #Show error if CSV generation fails

        #📊 Optional Numeric Column Chart (Plotly)
        numeric_cols = selected_df.select_dtypes(include=['int64', 'float64']).columns.tolist()  #Get numeric columns

        if numeric_cols:  # If numeric columns exist
            selected_col = st.selectbox("📈 Select Numeric Column for Distribution Chart", numeric_cols)  #Choose column

            #📊 Create histogram chart with full year ticks (if VisitYear is selected)
            if selected_col == 'VisitYear':
                #🧭 Sort years and make sure every year gets a tick (fix for even-year only display)
                sorted_years = sorted(selected_df[selected_col].dropna().unique())
                fig = px.histogram(
                    selected_df,
                    x=selected_col,
                    nbins=len(sorted_years),  #Set number of bins equal to number of unique years
                    title=f"Distribution of {selected_col}"
                )
                fig.update_layout(xaxis=dict(tickmode='array', tickvals=sorted_years))  #Force tick for each year
            else:
                fig = px.histogram(selected_df, x=selected_col, nbins=30, title=f"Distribution of {selected_col}")  #Default histogram

            st.plotly_chart(fig, use_container_width=True)  #Show chart


#SHORT DESCRIPTION
#This block enables users to explore, clean, visualize, and export various tourism-related datasets in a beginner-friendly UI

#🔍 DATA EXPLORER PAGE

#SHORT DESCRIPTION  
#This block enables users to explore, clean, visualize, and export various tourism-related datasets in a beginner-friendly UI

#🎯 PURPOSE  
#- Provide interactive raw data exploration  
#- Allow inspection of individual tables  
#- Enable data export as CSV for external analysis or reporting  
#- Provide visual summary (histograms)

#✨ KEY FEATURES IMPLEMENTED

#Interactive Page Selection  
# - elif page == "🔍 Data Explorer":     → Allows loading this block based on sidebar radio selection

#Section Heading & Instructions  
# - st.title(...)                        → Displays main heading on the app  
# - st.markdown(...)                     → Shows user guidance and info

#Load Multiple Dataset Views  
# - dataset_options = {...}              → Dictionary for label-to-DataFrame mapping  
# - st.selectbox(...)                    → Dropdown UI to let user select a dataset  
# - selected_df = dataset_options[...]   → Fetch the chosen DataFrame based on user input

#Data Cleaning  
# - df.drop_duplicates()                          → Removes duplicate rows  
# - df.apply(lambda col: col.fillna("Missing"))   → Fills missing object values with "Missing"

#Display Selected Data  
# - st.dataframe(selected_df)            → Shows DataFrame in interactive table layout

#Enable CSV Download  
# - df.to_csv(index=False).encode()      → Convert DataFrame to encoded CSV  
# - st.download_button(...)              → Streamlit button to download dataset as CSV

#Numeric Summary Chart  
# - df.select_dtypes()                   → Extract numeric columns  
# - st.selectbox() + px.histogram()      → User-selected column histogram display  

#Visualization  
# - px.histogram()                       → Plot distribution of a numeric column  
# - st.plotly_chart()                    → Embed the Plotly chart in the app  

#Enhanced Year-wise Histogram Support   
# - Handles special case for 'VisitYear' column  
# - Dynamically sets tick labels to show all years (e.g., 2013, 2014...) using:  
#   fig.update_layout(xaxis=dict(tickmode='array', tickvals=sorted_years))

#Best Practice Tip 
# - Use .dropna().unique() when setting custom tick labels to exclude NaNs  
# - Keeps axis labels clean and avoids plotting issues

#Error Handling  
# - try...except                        → Safe handling of CSV export failure  
# - st.warning()                        → Alert user for empty datasets  
# - st.error()                          → Show failure reasons if any

#📦 STREAMLIT COMMANDS USED  
# - st.title()                          → Adds a large title heading to the page  
# - st.markdown()                       → Renders text in Markdown format  
# - st.selectbox()                      → Creates a dropdown menu for selecting options  
# - st.dataframe()                      → Displays a scrollable, filterable table view of a DataFrame  
# - st.download_button()                → Adds a button to download files (e.g., CSV)  
# - st.plotly_chart()                   → Embeds Plotly charts inside the Streamlit app  
# - st.warning()                        → Warning for empty data  
# - st.error()                          → Error handling

#📊 PANDAS COMMANDS USED  
# - df.drop_duplicates()                → Removes duplicate rows from a DataFrame  
# - df.fillna()                         → Fills missing (null) values in a DataFrame  
# - df.apply(lambda ...)                → Applies a function across a column or row-wise  
# - df.select_dtypes()                  → Selects columns based on data types (e.g., numeric or object)  
# - df.to_csv(index=False)              → Converts DataFrame to a CSV string (used for downloads)

#📈 PLOTLY EXPRESS COMMANDS USED  
# - px.histogram()                      → Plots a histogram showing distribution of a numeric column  
# - fig.update_layout()                 → Used to update chart properties (like x-axis ticks)

#📃 PYTHON GENERAL  
# - encode('utf-8')                     → Converts text to bytes for downloading files (CSV format)  
# - if ... else ...                     → Conditional logic used for choosing how to handle null values  
# - dict{} and list[]                   → Used for mapping sheet names and dropdown options

# 📊 USER SUMMARY STATISTICS
elif page == "📊 User Summary Statistics":   #Condition to render the User Summary Statistics page

    #📌 Page Title
    st.title("📊 User Summary Statistics")   #Sets the page title in the Streamlit interface

    #🧭 Personalized Attraction Recommendations
    st.subheader("🧭 Personalized Attraction Suggestions")  #Subsection header for recommendation output

    #📝 Page Description
    st.markdown("""
    Get personalized travel attraction suggestions based on your preferences and past behavior.  
    Explore what similar users liked, discover trending attraction types, and view popular travel patterns by user segment.
    """)
    
    #Description explaining the logic and methods used for suggestions
    
    #This section provides **personalized attraction suggestions** using a hybrid approach:
      #- Collaborative filtering based on similar users
      #- Content-based filtering using attraction type and city
      #- Segment-based suggestions based on user preferences
    #It also highlights User Segment Trends and Popular Attraction Types to uncover tourism behavior patterns. 

    #📂 LOAD DATA: Updated Item Sheet
    file_path = r"C:\\Users\\Bala Sowntharya\\Downloads\\Tourism_Experience_Analytics_Dataset.xlsx"  #📁 Excel file path
    
    #🔄 Check if data already loaded in session (to avoid re-reading from disk)
    if 'Updated_Item' not in data:        #If 'Updated_Item' sheet is not already in the data dictionary
        
        #📥 Read 'Updated_Item' sheet from Excel
        updated_item_df = pd.read_excel(file_path, sheet_name='Updated_Item')
        
        #🧹 Strip leading/trailing spaces from column names
        updated_item_df.columns = updated_item_df.columns.str.strip()
        
        #🧼 Fill missing string values with 'Missing'
        updated_item_df = updated_item_df.apply(lambda col: col.fillna("Missing") if col.dtype == 'object' else col)
        #Replaces missing values in text columns with "Missing"
          #For each column in the DataFrame:
            # - If it's a text (object) column → fill NaN with "Missing"
            # - If it's numeric → leave it unchanged

    else:
        #♻️ If already loaded, use cached version
        updated_item_df = data['Updated_Item'] #Load from existing session data
    
    #Efficient Loading: Checks whether data is already loaded to avoid duplicate reads
    #Data Cleaning: Trims whitespace in column headers and fills missing text fields with "Missing"    

    #👥 USER SELECTION FOR RECOMMENDATION
    #🔍 Get unique, non-null User IDs
    user_ids = df_tx['UserId'].dropna().unique()  #Removes NaN and gets unique UserIds

    #🎛️ User dropdown in Streamlit to select a UserId
    selected_user = st.selectbox("👤 Select a User ID for Recommendations:", sorted(user_ids))  #Sorted dropdown of users

    #📊 Create User-Attraction interaction matrix
    user_attraction_matrix = df_tx.pivot_table(
        index='UserId',               #Rows = Users
        columns='AttractionId',       #Columns = Attractions
        aggfunc='size',               #Counts how many times each user visited each attraction
        fill_value=0                  #Fills missing interactions with 0
    )

    #dropna().unique() → Keeps only valid User IDs for dropdown
    #selectbox()       → Lets user pick a user for personalized recommendations
    #pivot_table()     → Builds interaction matrix used in collaborative filtering

    #Explanation:
      #This matrix helps build collaborative filtering logic 
         #(Users as rows, Attractions as columns, values = visit count or interaction)


    #🤝 User Simularity Calculation Using Cosine Simalarity

    #📦 Import cosine similarity function from sklearn
    from sklearn.metrics.pairwise import cosine_similarity  #Used to measure similarity between users based on their attraction visits
    
    #Why Cosine Similarity?
      # Cosine similarity measures the angle between two vectors (not their magnitude)
      # In this context:
        # - Each user is represented as a vector of interactions with attractions (1 = visited, 0 = not)
        # - Cosine similarity is ideal for sparse binary interaction data
        # - It captures user preference patterns even if total visit counts vary
        # - Value ranges from 0 (no similarity) to 1 (identical behavior)

    #🔢 Compute cosine similarity between users
    similarity_matrix = cosine_similarity(user_attraction_matrix)  #Returns a matrix of pairwise user similarities
    
    #📄 Convert similarity matrix to a DataFrame
    similarity_df = pd.DataFrame(
        similarity_matrix,                    #2D numpy array of similarity values
        index=user_attraction_matrix.index,   #Set row labels as User IDs
        columns=user_attraction_matrix.index  #Set column labels as User IDs
        )
    
    #How it works:
      # - user_attraction_matrix: Pivot table of (UserId x AttractionId), with 0/1 or visit counts
      # - similarity_df: Each (i,j) entry shows how similar User i is to User j
      # - Used to find users with similar tastes for collaborative filtering
    
    #🔎 Find Similar Users & Attractions (Collaborative Filtering Logic)
    #Step 1: Get top similar users
    similar_users = similarity_df[selected_user].sort_values(ascending=False)[1:6].index.tolist()
    
    #🔍 Explanation:
      # - similarity_df[selected_user] gives similarity scores between the selected user and all others
      # - sort_values(descending) ranks the most similar users first
      # - [1:6] skips the selected user (index 0) and picks top 5 most similar others
      # - This list will be used to explore what similar users liked

    #Step 2: Get attractions already visited by the selected user
    user_visited = set(df_tx[df_tx['UserId'] == selected_user]['AttractionId'])

    #🔍 Explanation:
      # - Filters transaction data for the selected user
      # - Converts their visited attraction list to a set for fast lookup
      # - Used to avoid re-recommending what the user already knows

    #Step 3: Get data for similar users
    similar_users_data = df_tx[df_tx['UserId'].isin(similar_users)]

    #🔍 Explanation:
      # - Filters transaction data for only the 5 similar users
      # - This data includes what they’ve visited (used for candidate recommendations)

    #Peer Data Filtering: Extracting and analyzing data only from users who are similar to the selected user
       # It means filtering the transaction data to include only records from similar users (top 5),
          #based on similarity scores (e.g., cosine similarity of user behavior)
      #👤 similar_users      → List of top 5 users most like the current user
      #🧾 df_tx              → The full transaction/visit history
      #🔍 similar_users_data → Only the visit data from the top similar users (peers)

    #Why It's Important:
      #Only want to look at what similar users did – not what everyone did
      #This helps identify attractions that are likely to interest the selected user

    #🔁 Analogy:
      # Imagine that we're recommending movies – We wouldn’t use everyone’s top picks,
         # but instead rely on people with similar taste. That’s peer data filtering!  

    #Step 4: Filter out attractions the user already visited
    similar_visited = similar_users_data[~similar_users_data['AttractionId'].isin(user_visited)]

    #🔍 Explanation:
      # - From the similar users’ visited list, remove attractions already known to selected user
      # - This gives potential new suggestions based on peer behavior

    #🎯 Purpose of this Block:
      # - Core of collaborative filtering
      # - Leverages peer behavior to find unexplored but relevant attractions

    #Top similar users	    Collaborative filtering assumes "similar users like similar things"
    #Avoiding duplicates	Recommending already-visited attractions hurts user experience
    #Peer data filtering	Pulls visit data from only relevant, similar users
    #Subtraction logic	    Isolates only new attractions for recommendation  

    #Finalize Recommendation Based on Peer Visits

    #Check if any new attractions were found from similar users
    if not similar_visited.empty:
        #📊 Rank attractions visited by similar users (but not yet by selected user)
        top_attractions = similar_visited['AttractionId'].value_counts().reset_index()
        top_attractions.columns = ['AttractionId', 'Score'] #Rename columns for clarity

        #🔍 Explanation:
          # - Counts how often each attraction appears in peer visits
          # - Higher count = more popular among similar users
          # - Used as a recommendation score (Score = popularity among similar users)

    else:
        #ℹ️ Fallback: No similar-user data found, use global popularity instead
        st.info("🤖 No similar-user recommendations found. Showing globally popular suggestions instead.")

        #📊 Global popularity (excluding attractions already visited by the user)
        global_visited = df_tx[~df_tx['AttractionId'].isin(user_visited)]
        top_attractions = global_visited['AttractionId'].value_counts().reset_index()
        top_attractions.columns = ['AttractionId', 'Score']  #Rename columns for consistency

        #🔍 Explanation:
          # - If collaborative filtering can't generate suggestions (no peer overlap),
            # we fall back to the most visited attractions overall
          # - This ensures recommendations are always available (fail-safe)

        #🎯 Purpose:
          # - Prioritize personalized suggestions from similar users
          # - Provide a reliable fallback to maintain user experience
          # - Score reflects either peer interest or global interest

    #Find Correct Column for Attraction Lookup
    #Check which column is available for attraction name: 'Attraction' or fallback to 'Name'
    col_name = 'Attraction' if 'Attraction' in updated_item_df.columns else ('Name' if 'Name' in updated_item_df.columns else None)
    
      #🔍 Explanation:
        # - Some datasets may label attraction names as 'Attraction' or 'Name'
        # - This line dynamically checks which one exists
        # - Returns the first match; if neither exists, returns None (for error handling)

      #🎯 Purpose:
        # - Makes code robust against inconsistent column naming in Excel/CSV files
        # - Avoids hardcoding and improves reusability across datasets
    
    #If Attraction Name Column Is Found, Merge & Rename
    if col_name:
        #🔗 Merge top attraction scores with attraction name info from Updated_Item sheet
        recommend_df = top_attractions.merge(
            updated_item_df[['AttractionId', col_name]], #Only bring necessary columns
            on='AttractionId', how='left')               #Use left join to retain all top recommendations
        
        #Rename the dynamic column (e.g., 'Attraction' or 'Name') to a consistent label
        recommend_df = recommend_df.rename(columns={col_name: "Attraction Name"})

        #Explanation:
          # - Merges recommendation scores with readable names for display
          # - Makes sure "Attraction Name" appears consistently, regardless of source column name

        #Purpose:
          # - Ensures recommendations are user-friendly (with names, not just IDs)
          # - Adds flexibility for datasets with different naming conventions

        #USER CONTROL: Number of Recommendations to Display
        top_n = st.slider("🔢 Select number of top recommendations to display:", 1, len(recommend_df), min(10, len(recommend_df)))
        #Minimum --> 1
        #Maximum --> Total available recommendations
        #Default --> 10 (or less if not enough data)
        
        #Display success message confirming selected count and user ID
        st.success(f"🎯 Top {top_n} Recommendations for User {selected_user}")
        
        #📊 Show the top N rows of the recommendation table
        st.dataframe(recommend_df.head(top_n))

        #Explanation:
          # - Uses Streamlit slider for interactivity
          # - Dynamically adjusts the number of displayed suggestions
          # - Helpful for debugging, user customization, or exploring more/less results

        #Purpose:
          # - Let user control how many suggestions they want to see
          # - Keeps UI clean and interactive

        #📥 EXPORT OPTION: Download Recommendations 
        #🧾 Choose file format (CSV or Excel)
        download_format = st.radio("📄 Download Recommendations As:", ['CSV', 'Excel'], horizontal=True)
        
        #📂 If CSV is selected
        if download_format == 'CSV':
            #Download button for CSV
            st.download_button("⬇️ Download CSV", recommend_df.head(top_n).to_csv(index=False).encode('utf-8'), f"User_{selected_user}_Recommendations.csv", mime="text/csv")
        else:
            #📁 Prepare Excel file using BytesIO
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                recommend_df.head(top_n).to_excel(writer, index=False, sheet_name=f'Recommendations_{selected_user}')
            #Download button for Excel
            st.download_button("⬇️ Download Excel", output.getvalue(), f"User_{selected_user}_Recommendations.xlsx", mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    
        #🎯 Purpose:
          #Allows users to download personalized recommendations for future use
          #Helpful in real-world deployment, reporting, and user sharing

        #Note:
          #CSV is lighter and easier for quick analysis
          #Excel is structured and allows multiple sheets or formatting
    
    #FALLBACK DISPLAY: If attraction name column not found
    else:
        #Show warning message to user
        st.warning("📝 Attraction name column not found in Updated_Item sheet. Showing only IDs.")
        #📊 Display raw attraction ID-based recommendations
        st.dataframe(top_attractions)

    #🎯 Purpose:
      # Acts as a fallback when the attraction name column is missing from the dataset
      # Ensures user still sees valid recommendation results (though less readable)

    #Note:
      # Handles edge cases like misnamed/missing column headers
      # Keeps the user informed while gracefully degrading functionality

    # 🎯 Attraction Suggestions Based on Similar Users
    #st.subheader("🎯 Attraction Suggestions Based on Similar Users")

    #👤 Step 1: User selection
    #user_id = st.selectbox("👤 Select a User ID for Recommendations:", df_user['UserId'].unique())

    #📍 Step 2: Get attractions visited by the selected user
    #visited = df_tx[df_tx['UserId'] == user_id]['AttractionId'].unique()

    #👥 Step 3: Find users who visited the same attractions (excluding selected user)
    #similar_users = df_tx[(df_tx['AttractionId'].isin(visited)) & (df_tx['UserId'] != user_id)]

    #🔢 Step 4: Get unique similar user IDs
    #similar_user_ids = similar_users['UserId'].unique()

    #🧾 Step 5: Fetch full visit data of those similar users
    #similar_visited = df_tx[df_tx['UserId'].isin(similar_user_ids)]

    #🆕 Step 6: Recommend attractions that similar users visited but current user hasn't
    #collab_recs = similar_visited[~similar_visited['AttractionId'].isin(visited)]['AttractionId'].value_counts().head(5)

    #📋 Step 7: Map those attraction IDs to their names/descriptions
    #collab_df = updated_item_df[updated_item_df['AttractionId'].isin(collab_recs.index)]

    #Purpose:
      # - This block implements a basic collaborative filtering approach
      # - Recommends attractions based on behavior of other users with similar interests
      # - Avoids re-suggesting already visited attractions

    #Insight:
      # - Unlike cosine similarity-based approaches, this logic uses simple intersection of visit history
      # - Effective and fast for small or medium datasets

    #Display collaborative filtering recommendations if available
    #if not collab_df.empty:
        #st.success("Top Recommendations from Similar Users:")
        #st.dataframe(collab_df[['AttractionId', 'Attraction']])

    #If no similar-user recommendations, use content-based filtering
    #else:
        #st.warning("🤖 No similar-user recommendations found. Showing content-based suggestions instead.")

        #Step 1: Get details of attractions the user already visited
        #visited_details = updated_item_df[updated_item_df['AttractionId'].isin(visited)]

        #Step 2: Extract unique AttractionTypeIds and CityIds from visited data
        #match_type_ids = visited_details['AttractionTypeId'].unique()
        #match_city_ids = visited_details['AttractionCityId'].unique()

        #📥 Step 3: Recommend attractions based on matched type or city (excluding already visited)
        #content_df = updated_item_df[
            #(~updated_item_df['AttractionId'].isin(visited)) &
            #(
                #updated_item_df['AttractionTypeId'].isin(match_type_ids) |
                #updated_item_df['AttractionCityId'].isin(match_city_ids)
            #)
        #].drop_duplicates(subset='AttractionId').head(5)

    #Logic Summary:
    # - If collaborative filtering has results: show top suggestions from similar users
    # - Else: recommend based on similar content — i.e., same type or city as what the user liked
    # - Ensures fallback suggestions always exist, improving user experience

        #Show content-based recommendations (if collaborative ones were not available)
        #if not content_df.empty:
        #    st.success("✅ Top Content-Based Recommendations:")
        #    st.dataframe(content_df[['AttractionId', 'Attraction', 'AttractionTypeId', 'AttractionCityId']])
        
        #Final fallback if nothing was found
        #else:
        #    st.error("No new attractions found among similar users or content-based criteria.")

        with st.expander("ℹ️ How These Recommendations Are Chosen"):
           st.markdown("""
            - We look at **users with similar travel history** to find attractions you might also enjoy.
            - We also suggest **places similar to what you’ve visited before** – by type or city.
            - If no match is found, we show some of the **most popular attractions** among all users.
        """)
       
        # - Collaborative Filtering: Finds other users with similar visit history and recommends places 
                                     #they liked that they haven’t visited yet
        # - Content-Based Filtering: Suggests places similar in type or city to the ones you've already visited
        # - Fallback to Popular Suggestions: If neither works, top visited places globally are shown
    

    # 👥 User Segment Labels Based on Visited Attractions
    st.subheader("👥 User Segment Labels")

    # 📥 Load necessary sheets
    #Load the sheets needed to analyze user visits and label segments
    df_tx = pd.read_excel(file_path, sheet_name="Transaction")              #🔁 Visit/transaction records
    updated_item_df = pd.read_excel(file_path, sheet_name="Updated_Item")   #🧾 Attraction metadata
    type_df = pd.read_excel(file_path, sheet_name="Type")                   #🏷️ Attraction type info

    # 🔗 Merge datasets for analysis
    #Step 1: Merge visit records with attraction details
    merged_df = df_tx.merge(updated_item_df, on='AttractionId', how='left')
    
    #Step 2: Merge with type info to include readable category labels
    merged_df = merged_df.merge(type_df, on='AttractionTypeId', how='left')

    #🔍 Purpose:
       # - Enrich transaction data with attraction names and type labels
       # - Enables user behavior analysis by segment (e.g., nature lover, historical explorer)

    #Identify top attraction type per user

    #🎯 Goal:
      #For each user, find the most frequently visited attraction type
      #This can be used to label users into segments (e.g., Nature Lover, Museum Enthusiast)

    #Step 1: Group by UserId and AttractionType to count visits
    top_type_per_user = merged_df.groupby(['UserId', 'AttractionType'])['AttractionId'].count().reset_index()

    #Step 2: Sort so that the most visited type for each user comes first
    top_type_per_user = top_type_per_user.sort_values(['UserId', 'AttractionId'], ascending=[True, False])

    #Step 3: Drop duplicates to keep only the top attraction type per user
    top_type_per_user = top_type_per_user.drop_duplicates(subset='UserId', keep='first')

    #Output:
      #A DataFrame with each user and their top visited attraction type (for segmentation)

    # 🏷️ Map to segments

    #Purpose: Convert specific attraction types into broader user segment labels for easier interpretation

    #🔁 Mapping dictionary: Defines how each AttractionType maps to a user segment
    type_to_segment = {
        'Nature & Wildlife Areas': 'Nature Enthusiast',
        'Beaches': 'Nature Enthusiast',
        'Water Parks': 'Adventure Seeker',
    }

    #Create a new column by mapping 'AttractionType' to user-friendly 'UserSegment'
    top_type_per_user['UserSegment'] = top_type_per_user['AttractionType'].map(type_to_segment)

    #Drop users with no matching segment (i.e., unmapped attraction types)
    top_type_per_user = top_type_per_user.dropna(subset=['UserSegment'])

    #Output: A DataFrame with each user’s top visited attraction type and their mapped user segment

    #Pie Chart - Segment Distribution 
    #🎯 Purpose:
      #Visualize the distribution of users across different behavior-based segments 
        #(e.g., Nature Enthusiasts, Adventure Seekers)

    #Step 1: Count users per segment    
    segment_counts = top_type_per_user['UserSegment'].value_counts().reset_index()
    segment_counts.columns = ['UserSegment', 'Count']

    #📊 Step 2: Create a donut-style pie chart using Plotly
    fig_pie = px.pie(segment_counts, values='Count', names='UserSegment',
                 title='📌 User Segment Distribution', hole=0.4)
    
    #Customize label display to show both percent and label
    fig_pie.update_traces(textinfo='percent+label')

    #Step 3: Display the chart in Streamlit
    st.plotly_chart(fig_pie)

    #Show only relevant columns (UserId and their assigned UserSegment)
    st.dataframe(top_type_per_user[['UserId', 'UserSegment']].reset_index(drop=True))
    #Display a table of users with their assigned behavioral segment (based on top attraction type)

    #📥 Download Segment Summary

    #Select download format (CSV or Excel)
    download_segment = st.radio("📄 Download Segment Data As:", ['CSV', 'Excel'], horizontal=True, key="segment_download")
    if download_segment == 'CSV':          #Provide CSV download button
        st.download_button(
           "⬇️ Download CSV", 
           top_type_per_user.to_csv(index=False).encode('utf-8'), 
           "User_Segment_Labels.csv", 
            mime="text/csv"
        )
    else:
         #Provide Excel download option
         output_seg = io.BytesIO()      #Temporary in-memory buffer for Excel file
         with pd.ExcelWriter(output_seg, engine='xlsxwriter') as writer:
              top_type_per_user.to_excel(writer, index=False, sheet_name='UserSegments')
         output_seg.seek(0)             #Move the pointer to the beginning after writing
         st.download_button(
             "⬇️ Download Excel", 
             output_seg.getvalue(), 
             "User_Segment_Labels.xlsx", 
              mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
         )

    #🎈 Explore Attractions by User Type
    st.subheader("🎈 Explore Attractions by User Type")

    #Combine UserId and Segment for display in dropdown
    user_options = top_type_per_user[['UserId', 'UserSegment']].apply(lambda row: f"{row['UserId']} - {row['UserSegment']}", axis=1)
    
    #User selection from dropdown
    selected_option = st.selectbox("Select a User ID with Segment", user_options)
    selected_user = int(selected_option.split(' - ')[0])
    user_segment = selected_option.split(' - ')[1]

    #🎯 Map segment to relevant attraction types
    if user_segment == 'Nature Enthusiast':
        relevant_types = ['Nature & Wildlife Areas', 'Beaches']
    elif user_segment == 'Adventure Seeker':
        relevant_types = ['Water Parks']
    else:
        relevant_types = []  #No types mapped

    #🔗 Get type IDs that match the segment-relevant types
    matching_type_ids = type_df[type_df['AttractionType'].isin(relevant_types)]['AttractionTypeId'].unique()

    #🎯 Filter attractions for recommendation
    recommended_attractions = updated_item_df[updated_item_df['AttractionTypeId'].isin(matching_type_ids)]

    #📋 Show recommended attractions for the user segment
    st.markdown(f"### 📌 Recommended Attractions for **{user_segment}**")
    st.dataframe(recommended_attractions[['Attraction', 'AttractionAddress']].reset_index(drop=True))

    #📊 ANALYSIS: Popular Attraction Types by Segment (Overall)
    st.subheader("🎯 Popular Attraction Types by Segment (Overall)")

    #Filter to only known segments from mapping
    segment_popularity = merged_df[merged_df['AttractionType'].isin(type_to_segment.keys())].copy()

    #Map AttractionType to Segment for analysis
    segment_popularity['UserSegment'] = segment_popularity['AttractionType'].map(type_to_segment)

    #📈 Count visits per type per segment
    popularity_count = segment_popularity.groupby(['UserSegment', 'AttractionType'])['AttractionId'].count().reset_index().rename(columns={'AttractionId': 'VisitCount'})

    #📊 Bar Chart - Most visited attraction types by segment
    fig_segment = px.bar(popularity_count, x='AttractionType', y='VisitCount', color='UserSegment', barmode='group', title="📊 Most Visited Attraction Types by User Segment", text='VisitCount')
    fig_segment.update_layout(xaxis_title="Attraction Type", yaxis_title="Visit Count", legend_title="User Segment")
    st.plotly_chart(fig_segment)

#📊User Summary Statistics Page Documentation 

#Short Description
#Provides personalized attraction recommendations, segments users by behavior, 
      #and visualizes popular attraction types interactively.

# 🎯 PURPOSE
# - Deliver customized attraction suggestions using hybrid techniques.
# - Classify users into behavior-based segments.
# - Analyze and visualize attraction preferences across segments.

# 🔄 WORKFLOW BREAKDOWN
# 1. Load necessary Excel sheets: Transaction, Updated_Item, and Type.
# 2. User selects User ID to generate personalized recommendations.
# 3. Create a user-attraction matrix and apply cosine similarity for collaborative filtering.
# 4. If collaborative suggestions are insufficient, use content-based filtering (type/city).
# 5. As a fallback, recommend globally popular attractions.
# 6. Identify top attraction type per user → Map to user segment.
# 7. Display segment-based trends and visualize using pie and bar charts.
# 8. Enable segment-wise attraction exploration and CSV/Excel download.

# ✨ Key Features Implemented
#Hybrid Recommendation System (Collaborative + Content-Based)
#User Segmentation based on Visit Behavior
#Segment-Specific Attraction Exploration
#Dynamic Pie and Bar Charts (Plotly)
#CSV and Excel Export for Recommendations and Segments

#Models & Technique Used (Overview)
# - Collaborative Filtering using cosine similarity from sklearn
# - Content-Based Filtering using city and attraction type match
# - Rule-Based Segmentation using top attraction type per user

#Models & Techniques Used (Indepth)

#Collaborative Filtering (User-Based)
# - Uses the User-Attraction matrix where rows = UserId and columns = AttractionId
# - Cosine similarity is applied to measure how similar each user is to others based on their visit history
# - Recommends attractions visited by similar users that the current user hasn't visited yet
# - Implemented using: cosine_similarity() from sklearn.metrics.pairwise

#Content-Based Filtering
# - If collaborative filtering returns few or no results, the system switches to content-based filtering
# - It identifies attractions that share similar characteristics with those the user has already visited:
#     (a) Same attraction types (AttractionTypeId)
#     (b) Same cities (AttractionCityId)
# - This method recommends places based on the **attributes of attractions**, not on other users' behavior

#Rule-Based User Segmentation
# - For each user, the most visited attraction type is calculated.
# - Based on this top type, users are tagged into custom segments:
#     • Nature & Wildlife Areas / Beaches → "Nature Enthusiast"
#     • Water Parks → "Adventure Seeker"
# - These segments are then used to:
#     • Show group-level trends
#     • Recommend popular attractions for that segment

#Tools & Libraries Used
# - pandas                  : Data handling and Manipulation
# - streamlit               : Web app/dashboard interface
# - plotly.express          : Creating dynamic and interactive charts
# - sklearn.metrics.pairwise: Collaborative filtering via cosine similarity
# - xlsxwriter + io         : Exporting data to downloadable Excel files

#Commands Used

#Pandas
#pd.read_excel()     : Read Excel sheets into DataFrames
#df.merge()          : Merge two DataFrames on a key column
#df.groupby()        : Group data for aggregation
#df.pivot_table()    : Create matrix format (UserId x AttractionId)
#df.fillna()         : Fill missing values
#df.dropna()         : Drop rows with missing values
#df.value_counts()   : Count frequency of values
#df.sort_values()    : Sort DataFrame by specific column
#df.drop_duplicates(): Remove duplicate rows
#df.to_csv()         : Convert DataFrame to CSV for download
#df.to_excel()       : Export DataFrame to Excel format

#📦 STREAMLIT
#st.title()          : Page title
#st.subheader()      : Section headers
#st.markdown()       : Add markdown-formatted text
#st.selectbox()      : Dropdown input for user selection
#st.slider()         : Slider input for top-N selection
#st.dataframe()      : Display DataFrame as an interactive table
#st.download_button(): Enable file download (CSV or Excel)
#st.radio()          : Radio buttons for format selection
#st.success(), st.warning(), st.info(), st.error(): Message boxes
#st.plotly_chart()   : Render Plotly charts
#st.expander()       : Collapsible section for explanation text

#📦 SKLEARN
#cosine_similarity(): Calculate user-to-user similarity matrix

#📦 PLOTLY.EXPRESS
#px.pie(): Create donut/pie charts for segment distribution
#px.bar(): Bar charts for top attraction types per segment

# 📦 XLSXWRITER + IO
#pd.ExcelWriter(): Write DataFrame to Excel format in memory
#io.BytesIO()    : In-memory binary stream for file download

# 📤 Output Sections
# 🧭 Personalized Attraction Suggestions → Top-N recommendations using collaborative/content-based filtering
# 🎯 Suggestions from Similar Users      → Attractions visited by similar users that target user hasn’t visited
# ✅ Content-Based Recommendations       → Recommendations based on matching type/city
# 👥 User Segment Labels                 → Segment users (e.g., Nature Enthusiast) based on attraction type
# 📌 Pie Chart                           → Segment distribution overview
# 🎈 Explore Attractions by User Type    → View attractions based on user’s assigned segment
# 📊 Popular Attraction Types by Segment → Bar chart showing top visited types by each user segment



# 📈 Analyze Trends
elif page == "📈 Analyze Trends": 

    #Page Title & Description
    st.title("📈 Explore Trends in Tourism Experiences")
    st.markdown("""
    Analyze tourism trends based on **user demographics**, **visit details**, and **attraction attributes**.
    Use the dropdown below to explore overall and year-specific trends.
    """)

    #📁 Load Required Libraries
    import pandas as pd                #For Data Manipulation
    import matplotlib.pyplot as plt    #For Plotting Charts
    import seaborn as sns              #For Styled Visualization

    #📂 Load Excel File and Sheets
    #📂 Set the file path to the Excel dataset (raw string used to handle backslashes)
    file_path = r"C:\\Users\\Bala Sowntharya\\Downloads\\Tourism_Experience_Analytics_Dataset.xlsx"

    #📄 Load the 'Transaction' sheet containing visit-level data
    df_transaction = pd.read_excel(file_path, sheet_name='Transaction')

    #👤 Load the 'User' sheet containing user demographic details
    df_user = pd.read_excel(file_path, sheet_name='User')

    #🏞️ Load the 'Updated_Item' sheet with attraction details
    df_updated_item = pd.read_excel(file_path, sheet_name='Updated_Item')

    #🧭 Load the 'Type' sheet specifying types of attractions
    df_type = pd.read_excel(file_path, sheet_name='Type')

    #🚗 Load the 'Mode' sheet representing modes of visit (e.g., Solo, Business)
    df_mode = pd.read_excel(file_path, sheet_name='Mode')

    #🏙️ Load the 'Cities' sheet with city identifiers and names
    df_cities = pd.read_excel(file_path, sheet_name='Cities')

    #🌍 Load the 'Countries' sheet with country IDs and names
    df_countries = pd.read_excel(file_path, sheet_name='Countries')

    #🗺️ Load the 'Region' sheet containing region information
    df_region = pd.read_excel(file_path, sheet_name='Region')

    #🌐 Load the 'Continent' sheet mapping continent data
    df_continent = pd.read_excel(file_path, sheet_name='Continent')


    #Merge and Prepare Dataset
    #Rename 'VisitMode' column to 'VisitModeId' in the transaction DataFrame for merging
    df_transaction.rename(columns={'VisitMode': 'VisitModeId'}, inplace=True)

    #🔗 Merge user data into transaction data using 'UserId'
    df = df_transaction.merge(df_user, on='UserId', how='left')

    #🔗 Merge attraction details using 'AttractionId'
    df = df.merge(df_updated_item, on='AttractionId', how='left')

    #🔗 Merge attraction type details using 'AttractionTypeId'
    df = df.merge(df_type, on='AttractionTypeId', how='left')

    #🔗 Merge visit mode descriptions using 'VisitModeId'
    df = df.merge(df_mode, on='VisitModeId', how='left')

    #🔗 Merge city names using 'CityId' (subset of city columns used)
    df = df.merge(df_cities[['CityId', 'CityName']], on='CityId', how='left')

    #🔗 Merge country names using 'CountryId'
    df = df.merge(df_countries[['CountryId', 'Country']], on='CountryId', how='left')

    #🔗 Merge region names using 'RegionId'
    df = df.merge(df_region[['RegionId', 'Region']], on='RegionId', how='left')

    #🔗 Merge continent names using 'ContinentId'
    df = df.merge(df_continent[['ContinentId', 'Continent']], on='ContinentId', how='left')

    #Remove rows where the 'Rating' value is missing
    df.dropna(subset=['Rating'], inplace=True)
 
    #🏷️ Rename columns for display consistency
    df.rename(columns={
        'VisitMode': 'VisitModeName',     #Visit mode label (e.g., Business)
        'AttractionName': 'Attraction',   #Attraction label
        'TypeName': 'AttractionType',     #Type of attraction
        'CityName': 'City'                #City label
    }, inplace=True)


    #🔢Year Selection

    #🔢 Extract a sorted list of unique years from the 'VisitYear' column, excluding NaN values
    all_years = sorted(df['VisitYear'].dropna().unique())

    #📅 Create a dropdown menu in Streamlit for selecting a year or viewing 'Overall'
    selected_year = st.selectbox("📅 Select Year to Analyze", options=['Overall'] + all_years)

    #🔍 Filter the DataFrame based on selected year (if not 'Overall')
    if selected_year != 'Overall':
        #📆 Filter the data to include only rows for the selected year
        df = df[df['VisitYear'] == selected_year]
        #📝 Display a heading to show the selected year context
        st.markdown(f"### 🔍 Showing trends for the year: **{selected_year}**")
    else:
        #📊 Display heading for overall trends (all years included)
        st.markdown(f"### 🔍 Showing **Overall Trends**")


    #📊 Region-Wise Visits

    #📊 Add a section title for the region-wise visit count chart
    st.subheader("🌍 Region-wise Visit Counts")
    #📈 Count the number of visits per region using value_counts() and reset the index to create a DataFrame
    region_df = df['Region'].value_counts().reset_index()
    #🏷️ Rename the resulting columns to 'Region' and 'Visit Count' for clarity
    region_df.columns = ['Region', 'Visit Count']
    #🎨 Create a new figure and axis object using matplotlib for plotting
    fig, ax = plt.subplots()
    #📊 Create a horizontal bar plot using seaborn to visualize visit counts by region
    sns.barplot(data=region_df, x='Visit Count', y='Region', palette='crest', ax=ax)
    #🏷️ Set the title of the bar chart
    ax.set_title("Visits by Region")
    #Display the generated plot within the Streamlit app
    st.pyplot(fig)

    #🧳Visit Modes

    #🧳 Add a section header for visit modes distribution
    st.subheader("🚗 Visit Modes Distribution")
    #📈 Count the number of visits by each visit mode and convert to a DataFrame
    visit_mode_df = df['VisitModeName'].value_counts().reset_index()
    #Rename the columns to 'Visit Mode' and 'Count' for clarity
    visit_mode_df.columns = ['Visit Mode', 'Count']
    #🎨 Create a matplotlib figure and axis for the bar chart
    fig, ax = plt.subplots()
    #📊 Create a vertical bar plot using seaborn to show the distribution of visit modes
    sns.barplot(data=visit_mode_df, x='Visit Mode', y='Count', palette='flare', ax=ax)
    #Set the chart title to describe the plot
    ax.set_title("Preferred Visit Modes")
    #Display the plot within the Streamlit app
    st.pyplot(fig)


    #🏝️ Attraction Types

    #🏝️ Add a section header for attraction type distribution
    st.subheader("🏝️ Attraction Type Distribution")
    #📈 Count the number of visits per attraction type and convert to a DataFrame
    attr_type_df = df['AttractionType'].value_counts().reset_index()
    #🏷️ Rename the columns to 'Attraction Type' and 'Visit Count' for readability
    attr_type_df.columns = ['Attraction Type', 'Visit Count']
    #🎨 Create a matplotlib figure and axis for the plot
    fig, ax = plt.subplots()
    #📊 Generate a horizontal bar plot to visualize visits by attraction type
    sns.barplot(data=attr_type_df, x='Visit Count', y='Attraction Type', palette='mako', ax=ax)
    #🏷️ Set a title for the chart
    ax.set_title("Visits by Attraction Type")
    #Display the plot in the Streamlit app
    st.pyplot(fig)


    #⭐ Average Ratings by Attraction Type

    #⭐ Add a section title to indicate average ratings by attraction type
    st.subheader("⭐ Average Ratings by Attraction Type")
    #📈 Group data by 'AttractionType', calculate the mean of 'Rating', and sort by rating (descending)
    avg_rating_df = df.groupby('AttractionType')['Rating'].mean().reset_index().sort_values(by='Rating', ascending=False)
    #🎨 Create a matplotlib figure and axis for the plot
    fig, ax = plt.subplots()
    #📊 Create a horizontal bar plot to show average ratings for each attraction type
    sns.barplot(data=avg_rating_df, x='Rating', y='AttractionType', palette='YlOrBr', ax=ax)
    #Set the title of the chart
    ax.set_title("Avg Ratings per Attraction Type")
    #Display the chart within the Streamlit app
    st.pyplot(fig)


    # 📅 Month-Wise Visit Distribution

    #📅 Add a section title for visualizing visit trends across months
    st.subheader("📆 Month-wise Visit Trends")
    #📈 Count the number of visits for each month, sort them in order (1–12), and reset the index
    month_df = df['VisitMonth'].value_counts().sort_index().reset_index()
    #Rename the columns to 'Month' and 'Visit Count' for better readability
    month_df.columns = ['Month', 'Visit Count']
    #🎨 Initialize the matplotlib figure and axis for plotting
    fig, ax = plt.subplots()
    #📊 Create a bar plot using seaborn to show visit counts per month
    sns.barplot(data=month_df, x='Month', y='Visit Count', palette='coolwarm', ax=ax)
    #Set a descriptive title for the chart
    ax.set_title("Visits by Month")
    #Display the plot inside the Streamlit app
    st.pyplot(fig)


    #📁 Download Option

    #📁 Add a section header for downloading the filtered dataset as a CSV file
    st.subheader("📥 Download Insights")
    #📦 Convert the final DataFrame to CSV format (no index) and encode it in UTF-8
    csv = df.to_csv(index=False).encode('utf-8')
    #🏷️ Generate a dynamic file name based on the selected year (e.g., 'Tourism_Trends_2023.csv' or 'Tourism_Trends_Overall.csv')
    file_label = f"Tourism_Trends_{selected_year}.csv" if selected_year != 'Overall' else "Tourism_Trends_Overall.csv"
    #📥 Add a Streamlit download button to allow users to download the CSV file
    st.download_button("📥 Download CSV", csv, file_name=file_label, mime='text/csv')

    #🧾 Summary
    #Scope     : Region, Country, Visit Mode, Attraction Type, Ratings, Month  
    #Data Shown: {'Selected Year' if selected_year != 'Overall' else 'All Years Combined'}  
    #Tools     : Pandas, Seaborn, Matplotlib, Streamlit  
    #Tip       : Use download option to analyze externally  

    #Short Description:
    #This module visualizes tourism trends across regions, visit modes, attraction types, and time (year/month)
    #It allows users to filter by year or view overall insights, with interactive charts and exportable data
 
    #🎯 PURPOSE:
    # - Understand tourism behavior patterns
    # - Identify peak visit times and popular regions/attractions
    # - Enable CSV download for further offline analysis

    #🧰 LIBRARIES & TOOLS USED:
    # - pandas           : Data handling
    # - matplotlib.pyplot: Plotting
    # - seaborn          : Styled graphs
    # - streamlit        : App Interface

    # 📚 PANDAS (pd)
    # - pd.read_excel()         : Load Excel sheet into DataFrame
    # - df.merge()              : Join multiple DataFrames on keys
    # - df.rename()             : Rename column headers
    # - df.dropna()             : Remove rows with missing values
    # - df.value_counts()       : Count unique occurrences
    # - df.groupby()            : Group by a column and perform aggregations
    # - df.sort_values()        : Sort DataFrame by a column
    # - df.to_csv()             : Export DataFrame to CSV format

    # 🖼️ MATPLOTLIB (plt)
    # - plt.subplots()          : Create figure and axes objects for plotting
    # - ax.set_title()          : Set title for the axes
    # - st.pyplot()             : Display matplotlib chart in Streamlit

    # 🌈 SEABORN (sns)
    # - sns.barplot()           : Create bar plots with statistical styling

    # 🧩 STREAMLIT (st)
    # - st.title()              : Set page title
    # - st.markdown()           : Add markdown-formatted descriptions
    # - st.selectbox()          : Dropdown selection UI
    # - st.subheader()          : Section headers
    # - st.pyplot()             : Render matplotlib charts
    # - st.download_button()    : Download processed data as CSV

    # 💡 OTHER
    # - sorted(), .unique()     : Native Python and NumPy functions used for dropdown options


#🧮 PREDICT VISIT MODE - Classification Page
elif page == "🧮 Predict Visit Mode":

    #Page Title
    st.title("🧮 Predict Visit Mode - Classification Model")  #Sets the title of the page

    #📝 Page Description
    st.markdown("""
    This section uses a classification model to **predict the travel mode** (e.g., Bus, Train) 
    based on user demographics and visit history.
    """)  #️⃣ Provides a short description about this section

    #📥 REQUIRED IMPORTS
    from sklearn.model_selection import train_test_split  #For splitting the dataset
    from sklearn.preprocessing import LabelEncoder        #For encoding target labels
    from sklearn.ensemble import RandomForestClassifier   #Random Forest for classification
    from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay  #Metrics
    import matplotlib.pyplot as plt                       #For visualizations

    #📦 Merge transaction and user data
    df = pd.merge(data['transactions'], data['user'], on='UserId', how='left')  #Merges transaction and user info
    
    #Function   : Joins transactions with user data using UserId  
    #Use Case   : To combine visit details and user demographics for model training  
    #Merge Type : left join keeps all transaction records, adds matching user info  

    #Drop rows with missing target labels
    df = df[df['VisitMode'].notna()]  #Keeps only rows where VisitMode is available

    #🎯 Define Features and Target
    X = df[['ContinentId', 'RegionId', 'CountryId', 'CityId', 'VisitYear', 'VisitMonth']]  #Input features
    y = df['VisitMode']                                                                    #Target variable

    #X --> Feature
    #y --> Target

    #Explanation:
    #Function     : Selects the input features (X) and the target label (y) for model training
    #Use Case     : Combines location and visit time data to predict user visit mode
    #X (Features) : ContinentId, RegionId, CountryId, CityId, VisitYear, VisitMonth
    #y (Target)   : VisitMode – The classification label (e.g., Business, Family, Solo)

    #Encode target labels
    le = LabelEncoder()              #Create Label Encoder instance to convert categories to numbers
    y_encoded = le.fit_transform(y)  #Encode VisitMode column to numeric labels (y → y_encoded)

    #Explanation:
    #Function   : Converts categorical VisitMode values into machine-readable numeric labels
    #Use Case   : Required because machine learning models (like Random Forest) need numeric input
    #Example    : "Family" → 0, "Business" → 1, etc. (based on internal alphabetical order)

    #Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)  #80-20 split
    
    #X, y_encoded,                 #Features and encoded target
    #test_size=0.2,                #20% of data used for testing, 80% for training
    #random_state=42               #Ensures reproducible results on each run
    
    #Explanation:
    #Function   : Splits the dataset into training and testing sets
    #Use Case   : Train the model on one part of the data and evaluate on unseen data
    #Split Type : 80% training, 20% testing (controlled via test_size=0.2)

    #📊 VISUALIZATION: Travel Mode Distribution in Raw Data
    st.markdown("### 📊 Travel Mode Distribution")  #Subheading for the chart

    #🔁 Mapping VisitMode codes to names
    visit_mode_mapping = {
        0: "-", 1: "Business", 2: "Couples", 3: "Family", 4: "Friends", 5: "Solo"
    }

    #0: "-",            #Placeholder or missing value
    #1: "Business",     #Business travel
    #2: "Couples",      #Couple trips
    #3: "Family",       #Family-oriented travel
    #4: "Friends",      #Friends group trips
    #5: "Solo"          #Solo travel

    #Explanation
    #Function   : Creates a mapping between encoded numeric VisitMode values and their actual names
    #Use Case   : Used for labeling charts and reports with readable visit mode names
    #Note       : Should match the encoded label order if used for decoding predictions

    #Map numeric VisitMode to readable names
    df['VisitModeName'] = df['VisitMode'].map(visit_mode_mapping)  #Creates a readable column for visualization

    #Explanation:
    #Function   : Creates a new column 'VisitModeName' with readable labels based on the numeric VisitMode codes
    #Use Case   : Enhances clarity in charts, summaries, and visualizations
    #Mapping    : Based on visit_mode_mapping dictionary (e.g., 2 → "Couples")

    #📊 Count each Visit Mode category
    visit_mode_counts = df['VisitModeName'].value_counts().reset_index() #Counts occurrences of each VisitModeName
    visit_mode_counts.columns = ['VisitMode', 'Count']                   #Rename columns for plotting

    #Explanation
    #Function   : Calculates how many times each visit mode appears in the dataset
    #Use Case   : Helps visualize mode distribution using bar charts

    #📈 Plot visit mode distribution using Plotly
    fig_mode = px.bar(
        visit_mode_counts,                    #Input data with VisitMode and Count
        x='VisitMode',                        #Categories shown on the x-axis
        y='Count',                            #Number of visits on the y-axis
        color='VisitMode',                    #Different colors for each visit mode
        title="Distribution of Visit Modes",  #Chart title
        text='Count'                          #Show count value on each bar
    )
    fig_mode.update_traces(textposition='outside')       #Display the count values outside each bar
    st.plotly_chart(fig_mode, use_container_width=True)  #Embed the Plotly chart in the Streamlit app

    #Explanation:
    #Function   : Visualizes the frequency of each Visit Mode in a colorful, interactive bar chart
    #Use Case   : Helps users quickly understand which travel modes are most common
    #Tool Used  : plotly.express (px) + Streamlit for web display

    #Model Training & Evaluation
    try:
        #🌲 Train Random Forest Classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)  #Build classifier with 100 trees
        model.fit(X_train, y_train)   #Fit model on training data (Train model on training dataset)

        #🔍 Predict on test data
        y_pred = model.predict(X_test)  #Predict on test dataset

        #Show Accuracy
        accuracy = accuracy_score(y_test, y_pred)       #Calculate accuracy (Measure how often predictions are correct)
        st.success(f" Model Accuracy: {accuracy:.2%}")  #Show accuracy

        #Define labels for confusion matrix
        display_labels = [visit_mode_mapping[i] for i in range(len(le.classes_))]  #Label mapping (Convert numeric labels to names)

        #📉 Show Confusion Matrix
        fig, ax = plt.subplots(figsize=(6, 4))  #Create plot
        disp = ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, display_labels=display_labels, ax=ax, cmap='viridis' #Plot confusion matrix with labels
        )

        #🎨 Add Colorbar
        if hasattr(ax.images[0], 'colorbar') and ax.images[0].colorbar is not None:
            ax.images[0].colorbar.set_label('Number of Records') #Label for matrix values
        else:
            fig.colorbar(ax.images[0], ax=ax).set_label('Number of Records') #Add colorbar manually if needed

        st.pyplot(fig)  #Show confusion matrix

        #Show Classification Report
        st.markdown("### Classification Report")  #Section title in UI
        report = classification_report(y_test, y_pred, target_names=display_labels)  #Text report (Precision, recall, F1-score)
        st.text(report)  #Render classification report

    except Exception as e:
        st.error(f" An error occurred while building the classification model: {e}")  #Handle and show error in app

    #Explanation:
    #Function   : Trains Random Forest model, evaluates performance, and shows confusion matrix & classification metrics
    #Use Case   : To classify visit modes with evaluation for accuracy and class-wise prediction strengths
    #Libraries  : sklearn (modeling, metrics), matplotlib (matrix plot), streamlit (UI output)


    #📊 VISUALIZATION: Predicted Visit Mode Year-wise
    st.markdown("### 📊 Year-wise Predicted Visit Modes")   #Section heading in Streamlit
 
    years = X_test['VisitYear'].reset_index(drop=True)#Extract year (Get VisitYear from X_test for alignment with predictions)
    predicted_modes = [visit_mode_mapping.get(i, "Unknown") for i in y_pred]  #Decode predicted labels

    year_mode_df = pd.DataFrame({
        'VisitYear': years,
        'PredictedVisitMode': predicted_modes
    }) #Combine years with predicted labels

    #📊 Group and count combinations
    grouped_counts = year_mode_df.groupby(['VisitYear', 'PredictedVisitMode']).size().reset_index(name='Count')  #Count per group
    #(Count how many times each mode was predicted per year)

    #📈 Bar Plot - Predicted Visit Mode Trends by Year
    fig_year_mode = px.bar(
        grouped_counts,
        x='VisitYear',
        y='Count',
        color='PredictedVisitMode',
        barmode='group',
        title='Predicted Travel Modes by Visit Year',
        category_orders={"VisitYear": sorted(year_mode_df['VisitYear'].unique())} #Sort year axis
    )
    #📤 Display plot in Streamlit
    st.plotly_chart(fig_year_mode, use_container_width=True) #Responsive layout for full-width display

    #Summary:
    #Function   : Visualizes how predicted travel modes vary across different years
    #Use Case   : Helps analyze yearly tourism trends and classifier behavior over time
    #Tools Used : plotly.express (interactive bar chart), pandas (groupby and count)

    #🧭 INTERACTIVE DROPDOWN: Filter by Year
    st.markdown("### 🗓️ Filter Travel Mode Predictions by Year")  #Section heading
    
    #📅 Unique years from predictions
    unique_years = sorted(year_mode_df['VisitYear'].unique())     #Get sorted list of unique years
    
    #🔽 Dropdown selector in sidebar or main area
    selected_year = st.selectbox("📅 Select Year", unique_years)  #Dropdown for year
    
    #🔍 Filter prediction data for selected year
    filtered_df = year_mode_df[year_mode_df['VisitYear'] == selected_year]  #Filter by selected year

    #📊 Count predicted visit modes in the selected year
    mode_counts = filtered_df['PredictedVisitMode'].value_counts().reset_index()
    mode_counts.columns = ['PredictedVisitMode', 'Count']          #Rename for plotting
    
    #📈 Bar chart: Mode prediction count for selected year
    fig_filtered = px.bar(
        mode_counts,
        x='PredictedVisitMode',
        y='Count',
        color='PredictedVisitMode',
        title=f'Predicted Visit Modes for {selected_year}'
    )

    #📤 Show plot in Streamlit
    st.plotly_chart(fig_filtered, use_container_width=True) #Full-width responsive chart

    #Summary:
    #Function   : Allows users to filter and view predicted travel mode distribution for a specific year
    #Use Case   : Enables detailed temporal analysis of classification results for targeted insights
    #Tools Used : Streamlit selectbox (dropdown), pandas filtering & counting, plotly.express bar chart

    #📆 INTERACTIVE DROPDOWN: Filter by Year & Month
    st.markdown("### 📆 Filter Travel Mode Predictions by Year & Month")  #Section title
    
    #Two-column layout for dropdowns
    selected_year_month = st.columns(2)  #Two-column layout for year & month

    #📅 Select Year
    with selected_year_month[0]:
        year_selected = st.selectbox("Select Year 📅", sorted(year_mode_df['VisitYear'].unique()), key="year_month")
    
    #📆 Select Month
    with selected_year_month[1]:
        month_selected = st.selectbox("Select Month 📆", sorted(X_test['VisitMonth'].unique()), key="month_select")
    
    #📋 Prepare DataFrame for filtering
    year_month_full_df = X_test[['VisitYear', 'VisitMonth']].reset_index(drop=True)
    year_month_full_df['PredictedVisitMode'] = predicted_modes  #Add predicted modes
    
    #🔎 Filter data for selected year & month
    filtered_month_df = year_month_full_df[
        (year_month_full_df['VisitYear'] == year_selected) &
        (year_month_full_df['VisitMonth'] == month_selected)
    ]  #️⃣ Filter by both year and month

    #📊 Count predictions
    month_mode_counts = filtered_month_df['PredictedVisitMode'].value_counts().reset_index()
    month_mode_counts.columns = ['PredictedVisitMode', 'Count']
   
    #📈 Plot bar chart for filtered results
    fig_month = px.bar(
       month_mode_counts,
       x='PredictedVisitMode',
       y='Count',
       color='PredictedVisitMode',
       title=f'Predicted Visit Modes for {year_selected}-{month_selected:02d}'
    )

    #🖼️ Display in Streamlit
    st.plotly_chart(fig_month, use_container_width=True)

    #Explanation
    #Function   : Enables year + month level filtering of predicted visit modes
    #Use Case   : Helps identify seasonal travel behavior trends in model predictions
    #Tools Used : Streamlit selectbox, pandas filtering, Plotly bar chart

#SHORT DESCRIPTION:
#This page builds a classification model to predict the mode of visit (e.g., Business, Family, Solo) 
     #using visitor demographic and transactional features
# It enables both model training and visual exploration of prediction results by year and month

#🎯 PURPOSE:
#To predict travel mode using classification (Random Forest) and visualize predictions with filters

#Workflow Breakdown
#1. Data Preparation	
    #Merged transactions with user data using UserId to combine travel behavior and user demographics

#2. Feature Engineering	
    #Selected input features: ContinentId, RegionId, CountryId, CityId, VisitYear, and VisitMonth
    #The target was VisitMode

#3. Label Encoding	
    #Transformed VisitMode (categorical) into numerical labels using LabelEncoder to prepare for model training

#4. Train-Test Split	
    #Divided the data into training (80%) and testing (20%) subsets for model validation

#5. Model Building	
#Trained a Random Forest Classifier with 100 estimators to learn visit mode patterns

#Why Random Forest?
  # - Handles both categorical and numerical data efficiently
  # - Robust to noise and overfitting due to ensemble of trees
  # - Requires minimal preprocessing (no need for scaling or normalization)
  # - Works well with tabular datasets and can handle missing or less-informative features
  # - Provides feature importance for interpretability

#6. Evaluation Metrics	
    #Displayed model accuracy, a confusion matrix, and a classification report (precision, recall, F1-score)

#7. Distribution Visualization	
    #Showed actual visit mode counts using Plotly bar charts to understand data skew and balance

#8. Year-wise Prediction Trends	
    #Visualized predicted visit modes across years to detect long-term tourism trends

#9. Year Filter (Interactive)	
    #Enabled filtering of predictions by selected year to inspect prediction distribution

#10. Year-Month Filter (Interactive)	
    #Further drilled down predictions by year and month to analyze seasonal trends and behaviors

#✨ KEY FEATURES IMPLEMENTED

#Predicts visit mode using machine learning
#Interprets model performance using classification metrics
#Interactive filtering by Year and Month
#Visualizes results with responsive and colorful charts
#Suitable for trend analysis, tourism behavior study, and personalization

#Interactive Page Selection
# - elif page == "🧮 Predict Visit Mode"      → Loads this logic on sidebar selection

#📊 Data Preparation & Cleaning
# - pd.merge()                                → Merge transaction and user datasets
# - df[df['VisitMode'].notna()]               → Remove rows with missing labels

#🔍 Classification Model
# - LabelEncoder()                            → Encode visit modes to numeric labels
# - RandomForestClassifier()                  → Fit and predict classification model
# - accuracy_score()                          → Show model accuracy
# - classification_report()                   → Display precision, recall, F1

#📊 Visualizations
# - px.bar()                                  → Plotly bar charts for mode counts
# - ConfusionMatrixDisplay                    → Show confusion matrix

#📅 Dynamic Filtering
# - st.selectbox()                            → Filter by Year and Month
# - grouped DataFrame                         → Aggregated prediction counts

#Tools & Libraries Used

#Machine Learning: 
#RandomForestClassifier, train_test_split, LabelEncoder, accuracy_score, classification_report, ConfusionMatrixDisplay
#Data Processing: pandas, numpy
#Visualization: matplotlib, plotly.express, Streamlit
#UI: st.selectbox, st.columns, st.markdown, st.plotly_chart, st.pyplot, st.success

#Commands Used (Overview)
#pandas                                     - For data manipulation (merge, filter, groupby, value_counts, etc.)
#sklearn.model_selection.train_test_split   - To split data into training and testing sets
#sklearn.preprocessing.LabelEncoder         - To convert categorical VisitMode labels into numeric values
#sklearn.ensemble.RandomForestClassifier    - To build and train a Random Forest classification model
#sklearn.metrics                            - For evaluating model performance (accuracy, confusion matrix, report)
#matplotlib.pyplot & ConfusionMatrixDisplay - To display the confusion matrix with colorbar
#plotly.express                             - To create interactive bar charts for visit mode analysis & predictions
#streamlit                                  - To build interactive UI (dropdowns, charts, layout, text outputs)

#Commands Used
#📦 STREAMLIT
# - st.title()                       → Add title to the page
# - st.markdown()                    → Description and subheadings
# - st.success()                     → Show success messages (accuracy)
# - st.error()                       → Show error messages if any
# - st.pyplot()                      → Display matplotlib plots
# - st.plotly_chart()                → Show Plotly charts
# - st.selectbox()                   → Dropdowns for filtering

#📊 PANDAS
# - pd.merge()                       → Merge multiple datasets
# - df['col'].notna()                → Filter non-null values
# - df.groupby().size().reset_index()→ Group and count predictions
# - df.value_counts()                → Count categories
# - pd.DataFrame()                   → Create new dataframes

#🤖 SKLEARN
# - train_test_split()               → Split into train/test sets
# - LabelEncoder()                   → Convert categorical to numeric
# - RandomForestClassifier()         → Classification model
# - accuracy_score()                 → Accuracy metric
# - classification_report()          → Full evaluation metrics
# - ConfusionMatrixDisplay           → Confusion matrix plotting

#🎨 MATPLOTLIB & PLOTLY
# - plt.subplots()                   → Create subplot for matrix
# - fig.colorbar()                   → Add color bar to matrix
# - px.bar()                         → Create bar charts

#Page Overview Summary
# - Goal              : Predict user visit mode using classification based on demographics and visit timing
# - Model Used        : Random Forest Classifier (ensemble of decision trees)
# - Input Features    : ContinentId, RegionId, CountryId, CityId, VisitYear, VisitMonth
# - Target Variable   : VisitMode (e.g., Business, Family, Couples, Friends, Solo)
# - Performance Output: Displays model accuracy, classification report, and confusion matrix
# - Visualizations:
#     - Raw Visit Mode distribution (bar chart)
#     - Year-wise predicted visit mode trends
#     - Dynamic filters for Year and Month to view custom prediction breakdown


#🎯Check if the selected page is "📈 Predicted Ratings" and render this section
elif page == "📈 Predict Ratings":
    #Page Title & Description
    st.markdown("## 📈 Predict Tourist Attraction Ratings")
    #Display the main page title with a larger markdown header
    st.markdown("""
    This page uses advanced data patterns to **predict how users might rate tourist attractions**  
    based on their preferences, visit details, and attraction features.  
    This helps platforms **suggest better attractions** and **improve service quality**.
    """)

    # 📂 Load Excel Sheets
    # 📁 Define the local path to the Excel dataset
    file_path = r"C:\Users\Bala Sowntharya\Downloads\Tourism_Experience_Analytics_Dataset.xlsx"
    #📄 Load the 'Transaction' sheet containing user visits, attraction IDs, and ratings
    df_transaction = pd.read_excel(file_path, sheet_name='Transaction')

    # 👤 Load the 'User' sheet containing user demographics
    df_user = pd.read_excel(file_path, sheet_name='User')

    # 🗺️ Load the 'Updated_Item' sheet containing detailed attraction information
    df_updated_item = pd.read_excel(file_path, sheet_name='Updated_Item')

    # 🏞️ Load the 'Type' sheet to get types of attractions (e.g., museum, beach)
    df_type = pd.read_excel(file_path, sheet_name='Type')

    # 🚗 Load the 'Mode' sheet to get travel or visit modes (e.g., family, solo)
    df_mode = pd.read_excel(file_path, sheet_name='Mode')

    # 🏙️ Load the 'Cities' sheet to map CityId to CityName
    df_cities = pd.read_excel(file_path, sheet_name='Cities')

    # 🌍 Load the 'Countries' sheet to map CountryId to Country names
    df_countries = pd.read_excel(file_path, sheet_name='Countries')

    # 🌐 Load the 'Region' sheet to map RegionId to regional names
    df_region = pd.read_excel(file_path, sheet_name='Region')

    # 🗺️ Load the 'Continent' sheet to map ContinentId to continent names
    df_continent = pd.read_excel(file_path, sheet_name='Continent')

    #Preprocessing
    # 🔄 Rename 'VisitMode' column to 'VisitModeId' to match with other sheets
    df_transaction.rename(columns={'VisitMode': 'VisitModeId'}, inplace=True)

    # 🔗 Merge user demographic data using UserId
    df = df_transaction.merge(df_user, on='UserId', how='left')

    # 🔗 Merge attraction details using AttractionId
    df = df.merge(df_updated_item, on='AttractionId', how='left')

    # 🔗 Merge attraction type (e.g., Nature, Religious) using AttractionTypeId
    df = df.merge(df_type, on='AttractionTypeId', how='left')

    # 🔗 Merge visit mode (e.g., Business, Family) using VisitModeId
    df = df.merge(df_mode, on='VisitModeId', how='left')

    # 🏙️ Merge city names using CityId (only keeping CityId and CityName columns)
    df = df.merge(df_cities[['CityId', 'CityName']], on='CityId', how='left')

    # 🌍 Merge country names using CountryId
    df = df.merge(df_countries[['CountryId', 'Country']], on='CountryId', how='left')

    # 🌐 Merge region names using RegionId
    df = df.merge(df_region[['RegionId', 'Region']], on='RegionId', how='left')

    # 🗺️ Merge continent names using ContinentId
    df = df.merge(df_continent[['ContinentId', 'Continent']], on='ContinentId', how='left')

    #Remove rows where Rating is missing (essential for supervised regression)
    df.dropna(subset=['Rating'], inplace=True)

    # 🔍 Feature Selection
    features = ['UserId', 'Region', 'Country', 'Continent', 'Attraction', 'VisitMode',
                'CityName', 'VisitYear', 'VisitMonth', 'AttractionType']
    #Define the list of input features to be used for rating prediction
    #These include user info, geographic info, attraction metadata, and time-based features

    # 🔍 Feature Selection
    #Purpose:
    # Select the relevant input features that influence user ratings, based on user demographics,
    # travel behavior, and attraction details.

    #Features Selected:
    # - UserId         → Unique identifier for each user (personal history tracking)
    # - Region         → Regional classification of the attraction
    # - Country        → Country where the attraction is located
    # - Continent      → Continent of the attraction
    # - Attraction     → Name or ID of the tourist attraction
    # - VisitMode      → Purpose or mode of visit (e.g., Business, Family, Couples)
    # - CityName       → City in which the attraction is located
    # - VisitYear      → Year when the visit occurred (temporal pattern)
    # - VisitMonth     → Month of the visit (seasonal effects)
    # - AttractionType → Category/type of the attraction (e.g., Historical, Adventure)

    #These features provide a mix of:
    # - Geographic context (City, Country, Region, Continent)
    # - Temporal signals (VisitYear, VisitMonth)
    # - User behavior (UserId, VisitMode)
    # - Attraction characteristics (Attraction, AttractionType)

    # 🔧 Extract these columns into a new DataFrame for modeling
    features = ['UserId', 'Region', 'Country', 'Continent', 'Attraction', 'VisitMode',
                'CityName', 'VisitYear', 'VisitMonth', 'AttractionType']


    df_model = df[features + ['Rating']].copy()
    #Create a new DataFrame for modeling by selecting only the chosen features and the target variable 'Rating'
    #Use .copy() to avoid modifying the original DataFrame by reference

    # 🔁 Target Encoding

    # 🔃 Loop through each categorical feature to encode using mean rating
    for col in ['UserId', 'Region', 'Country', 'Continent', 'Attraction', 'VisitMode', 'CityName', 'AttractionType']:
        # 📊 Calculate the mean rating for each unique value in the categorical column
        means = df_model.groupby(col)['Rating'].mean()
        # Map the mean rating back to the original dataframe as a new encoded feature (e.g., 'UserId_enc')
        df_model[col + '_enc'] = df_model[col].map(means)

    # 📦 Final Feature Set
    X = df_model[[col + '_enc' for col in ['UserId', 'Region', 'Country', 'Continent',
                                           'Attraction', 'VisitMode', 'CityName', 'AttractionType']]
                 + ['VisitYear', 'VisitMonth']]
    #Select the final input features for the model:
    #Encoded categorical variables (e.g., 'UserId_enc', 'Region_enc', etc.)
    #Numerical time-based features ('VisitYear', 'VisitMonth')

    y = df_model['Rating']
    # 🎯 Define the target variable – user rating of the attraction
    
    # 🔀 Train-Test Split
    # 📚 Import the train_test_split function from scikit-learn
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #Split the data into training and testing sets
    #80% for training, 20% for testing
    #random_state=42 ensures reproducibility of the split

    # ⚡ Train XGBoost Regressor
    # 📦 Import the XGBoost library for regression modeling
    import xgboost as xgb
    # 📊 Import evaluation metrics to assess model performance
    from sklearn.metrics import r2_score, mean_squared_error

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    #Initialize the XGBoost Regressor with the following parameters:
    # 🔹 n_estimators: number of boosting rounds (trees)
    # 🔹 learning_rate: step size shrinkage to prevent overfitting
    # 🔹 max_depth: maximum depth of each tree
    # 🔹 subsample: fraction of samples used per tree (helps generalization)
    # 🔹 colsample_bytree: fraction of features used per tree
    # 🔹 random_state: ensures reproducibility

    #Train the XGBoost model using the training data
    model.fit(X_train, y_train)

    # 🎯 Make predictions on the test data
    y_pred = model.predict(X_test)

    # 📊 Evaluation
    r2 = r2_score(y_test, y_pred)
    # 📈 Calculate R² Score – measures how well predictions approximate actual values
    mse = mean_squared_error(y_test, y_pred)
    # 🧮 Calculate Mean Squared Error – penalizes larger errors more heavily

    # 📝 Display a subheader on the Streamlit app for the evaluation section
    st.subheader("📉 Model Evaluation Metrics")
    eval_df = pd.DataFrame({
        "Metric": ["R² Score", "Mean Squared Error"],
        "Value": [f"{r2 * 100:.2f}%", f"{mse:.2f}"]
    })
    # 📊 Create a DataFrame to neatly format and display evaluation metrics

    st.table(eval_df)
    # 📋 Render the evaluation metrics table in the Streamlit app

    # 📉📈 Graph - Actual vs Predicted
    import matplotlib.pyplot as plt  # 🖼️ Import Matplotlib for plotting
    import seaborn as sns            # 🌈 Import Seaborn for enhanced statistical visualizations

    # 📝 Add a Streamlit subheader for the prediction comparison plot
    st.subheader("📈 Predicted vs Actual Ratings")

    # 📐 Create a Matplotlib figure and axes with a specific size
    fig, ax = plt.subplots(figsize=(6, 4))


    sns.scatterplot(x=y_test, y=y_pred, ax=ax, color="teal", s=50)
    # 🔹 Create a scatter plot with actual ratings on x-axis and predicted ratings on y-axis
    # 🔸 Use teal color and size 50 dots for better visibility

    ax.set_xlabel("Actual Ratings")             # 🏷️ Set label for x-axis
    ax.set_ylabel("Predicted Ratings")          # 🏷️ Set label for y-axis
    ax.set_title("Predicted vs Actual Ratings") # 🏷️ Set the chart title
    st.pyplot(fig) # 📊 Render the Matplotlib figure inside the Streamlit app

    # 📁 Sample Predictions Table
    st.subheader("📄 Sample Predictions Table")  #📝 Display a subheader for the sample output section
    sample_df = pd.DataFrame({'Actual': y_test[:10].values, 'Predicted': y_pred[:10]})
    # 📋 Create a DataFrame with the first 10 actual and predicted values
    # 🔹 Useful for quick comparison and debugging
    st.dataframe(sample_df)  # 📊 Display the sample predictions as an interactive table in Streamlit

    # 💾 Download CSV option for predictions
    import io              # 📦 Import io module to handle in-memory text streams for download
    csv = sample_df.to_csv(index=False).encode('utf-8')
    # 🧾 Convert the sample predictions DataFrame to a CSV format (no index)
    # 🔐 Encode the CSV string to UTF-8 for compatibility
    st.download_button(
        label="📥 Download Predictions as CSV",
        data=csv,
        file_name='predicted_vs_actual_ratings.csv',
        mime='text/csv',
    )
    # ⬇️ Display a download button in the Streamlit app
    # 📎 Allows user to download the predictions as a CSV file
    # 🗂️ Filename is 'predicted_vs_actual_ratings.csv', format is 'text/csv'

    # 💾 Download CSV option for evaluation
    eval_csv = eval_df.to_csv(index=False).encode('utf-8')
    # 🧾 Convert the evaluation metrics DataFrame to CSV (exclude index)
    # 🔐 Encode to UTF-8 for proper CSV formatting and compatibility
    st.download_button(
        label="📥 Download Evaluation Metrics",
        data=eval_csv,
        file_name='model_evaluation_metrics.csv',
        mime='text/csv',
    )
    # ⬇️ Create a download button in the Streamlit app
    # 📎 Enables users to download evaluation results as a CSV file
    # 📂 Filename: 'model_evaluation_metrics.csv', MIME type: text/csv

#🔄 PAGE TRIGGER:
# elif page == "📈 Predicted Ratings" → Executes this logic when selected from the sidebar

# 🎯 Purpose:
# Predict how users might rate tourist attractions using machine learning (XGBoost)  
# - Help platforms improve service quality by understanding predicted satisfaction  
# - Enable pre-emptive enhancement of lower-rated experiences based on predicted feedback  

#🌐 USE CASES:
# - Personalize attraction suggestions based on predicted user preferences
# - Identify attractions with potential for negative reviews
# - Strengthen marketing strategies by anticipating satisfaction trends

# 🔄 Workflow Breakdown:
# 1. Data Loading & Merging:
#    - Loaded multiple Excel sheets: Transaction, User, Item, Location Hierarchy
#    - Merged to form a unified dataframe with attraction + user + geo + visit info

# 2. Preprocessing:
#    - Renamed columns for clarity (e.g., VisitMode → VisitModeId)
#    - Dropped rows with missing Ratings

# 3. Feature Engineering:
#    - Selected features: Region, Country, Continent, Attraction, VisitMode, etc.
#    - Used Target Encoding to numerically represent categorical variables

# 4. Model Training:
#    - Trained an XGBoost Regressor with advanced tuning
#    - Used 80/20 train-test split for evaluation

# 5. Predictions & Evaluation:
#    - Compared predicted vs actual ratings
#    - Visualized results using scatter plots
#    - Tabulated evaluation metrics (R², MSE)
#    - Offered downloadable outputs for predictions and evaluations

# ✨ KEY FEATURES IMPLEMENTED:
# - XGBoost model for robust predictions
# - Target Encoding of categorical features
# - Evaluation table with R² Score and MSE
# - Interactive scatter plot for predictions vs actuals
# - Download buttons for evaluation metrics & predictions

# 🤖 MODELS & TECHNIQUES USED:
# - XGBoost Regressor → Used for accurate numeric prediction
# - Target Encoding   → Converts categorical features using mean of target variable
# - Train-test split  → Ensures model is evaluated on unseen data

# 🛠️ TOOLS & LIBRARIES USED:
# Streamlit
# - st.markdown(), st.subheader(), st.table(), st.dataframe()  → For layout & display
# - st.download_button(), st.pyplot()                          → For downloads & plotting

# Pandas
# - pd.read_excel(), merge(), to_csv(), groupby() → Data wrangling and export

# Sklearn
# - train_test_split, r2_score, mean_squared_error → ML utilities

# XGBoost
# - xgb.XGBRegressor → Model implementation

# Matplotlib / Seaborn
# - plt.subplots(), sns.scatterplot() → Visualization

# 📘 STREAMLIT COMMANDS USED:
# - st.markdown()          → For adding headers and descriptions
# - st.subheader()         → Section titles
# - st.table(), st.dataframe() → Tabular data outputs
# - st.download_button()   → CSV download for outputs
# - st.pyplot()            → Render plots in Streamlit

# 📊 PANDAS COMMANDS USED:
# - pd.read_excel()        → Load individual sheets
# - df.merge()             → Combine datasets
# - df.dropna()            → Clean missing values
# - df.groupby()           → Aggregate ratings for target encoding
# - df.to_csv()            → Prepare CSV for download

# 📈 SKLEARN & XGBOOST COMMANDS:
# - train_test_split()     → Split data for evaluation
# - r2_score(), mean_squared_error() → Model evaluation metrics
# - xgb.XGBRegressor()     → Model used for regression

# 🧭 PAGE SUMMARY:
# - Goal               : Predict user satisfaction (ratings) for attractions  
# - Datasets Used      : User, Transaction, Updated_Item, Mode, Region, etc.  
# - Features Used      : Categorical (encoded) + Visit year/month  
# - Model Used         : XGBoost Regressor  
# - Output             : Predicted ratings, evaluation summary, visuals, export options  

# 📊 OUTPUT SECTIONS:
# 1. Model Evaluation Table (R², MSE)
# 2. Actual vs Predicted Scatter Plot
# 3. Sample Predictions Table
# 4. Download CSV for Predictions
# 5. Download CSV for Evaluation Metrics


#🌍 Get Recommendations
elif page == "🌍 Get Recommendations":    #Triggers this section when 'Get Recommendations' is selected in the sidebar

    #PAGE TITLE & DESCRIPTION
    st.title("🌍Get Recommendations")     ##Sets the title for the page
    st.markdown("""
    This section provides interactive **region, country, and continent-wise visit insights** across different years.  
    It enables exploration of geographic trends using historical tourism data.
    """) 
    #Use Case: Helps users understand they will explore visit patterns by location and year.
    
    # 📆 LOAD DATA
    @st.cache_data(show_spinner=False) #Caches function output to avoid reloading on every run and disables loading spinner
    def load_data():                   #Function to load and clean all Excel sheets
        file_path = r"C:\\Users\\Bala Sowntharya\\Downloads\\Tourism_Experience_Analytics_Dataset.xlsx" #📂 Specifies the path to the Excel file
        dfs = {                                                               #Dictionary to load and store all sheets
            'User': pd.read_excel(file_path, sheet_name='User'),              #Loads 'User' sheet into dataframe
            'Country': pd.read_excel(file_path, sheet_name='Countries'),      #Loads 'Countries' sheet
            'Region': pd.read_excel(file_path, sheet_name='Region'),          #Loads 'Region' sheet
            'Continent': pd.read_excel(file_path, sheet_name='Continent'),    #Loads 'Continent' sheet
            'Transaction': pd.read_excel(file_path, sheet_name='Transaction') #Loads 'Transaction' sheet
        }
        for key in dfs:                                                       #Loop over each dataframe in the dictionary
            #Removes extra spaces from column names to avoid key errors
            dfs[key].columns = dfs[key].columns.str.strip()
            #Fills missing values in categorical columns with 'Missing'
            dfs[key] = dfs[key].apply(lambda col: col.fillna("Missing") if col.dtype == 'object' else col)
        return dfs #Returns the cleaned data dictionary

    data = load_data()               #Calls the load_data function and stores all sheets
    df_user = data['User']           #Extracts 'User' dataframe
    df_country = data['Country']     #Extracts 'Country' dataframe
    df_region = data['Region']       #Extracts 'Region' dataframe
    df_continent = data['Continent'] #Extracts 'Continent' dataframe
    df_tx = data['Transaction']      #Extracts 'Transaction' dataframe

    # 📊 Overall Region/Country/Continent Analysis
    st.subheader("🌐 Region, Country & Continent - Overall Visit Counts") #Displays a subheader on the Streamlit page

    # 🔹 Region-wise Overall
    if 'RegionId' in df_user.columns:
        #Checks if 'RegionId' is directly available in the user data
        df_tx_region = df_tx.merge(df_user[['UserId', 'RegionId']], on='UserId', how='left') #Merge transaction data with RegionId from user data
    else:
        df_tx_region = df_tx.merge(df_user[['UserId', 'CountryId']], on='UserId', how='left') #Merge transaction data with CountryId (if RegionId missing)
        df_tx_region = df_tx_region.merge(df_country[['CountryId', 'RegionId']], on='CountryId', how='left') #Then merge with country table to get RegionId

    df_tx_region = df_tx_region.merge(df_region[['RegionId', 'Region']], on='RegionId', how='left') #Final merge to get Region names from RegionId
    region_counts = df_tx_region['Region'].value_counts().reset_index()
    #Count total visits per region and reset index for plotting
    region_counts.columns = ['Region', 'VisitCount']
    #Rename columns to 'Region' and 'VisitCount'

    fig_region = px.bar(region_counts,              #📊 Data containing regions and their visit counts
                        x='Region',                 #🪧 X-axis shows Region names
                        y='VisitCount',             #📈 Y-axis shows number of visits
                        color='VisitCount',         #🎨 Color intensity based on visit count
                        color_continuous_scale='viridis', title='Overall Region-wise Visits') #🌈 Viridis color scale for better contrast
    st.plotly_chart(fig_region, use_container_width=True)     #Renders the Plotly bar chart in the Streamlit app using full container width                                

    #Country-wise Overall
    if 'RegionId' in df_user.columns:
        #🔍 Check if RegionId is directly present in the User sheet
        df_tx_full = df_tx.merge(df_user[['UserId', 'CountryId', 'RegionId']], on='UserId', how='left')
        #🔗 Merge transaction with both CountryId and RegionId from user data
    else:
        df_tx_full = df_tx.merge(df_user[['UserId', 'CountryId']], on='UserId', how='left')
        #🔗 Merge transaction with CountryId (if RegionId not directly available)
        df_tx_full = df_tx_full.merge(df_country[['CountryId', 'RegionId']], on='CountryId', how='left')
        #🔗 Retrieve RegionId via Country table if missing in user data

    df_tx_full = df_tx_full.merge(df_country[['CountryId', 'Country']], on='CountryId', how='left')
    #🗺️ Add actual Country names by joining with Country table
    df_tx_full = df_tx_full.merge(df_region[['RegionId', 'Region', 'ContinentId']], on='RegionId', how='left')
    #🧭 Merge to get Region names and ContinentId from Region table
    df_tx_full = df_tx_full.merge(df_continent[['ContinentId', 'Continent']], on='ContinentId', how='left')
    #🌍 Final merge to get Continent names from ContinentId

    country_counts = df_tx_full['Country'].value_counts().reset_index()
    #📊 Count total visits per country and convert to DataFrame
    country_counts.columns = ['Country', 'VisitCount']
    #📝 Rename the columns for better readability

    fig_country = px.bar(country_counts,                       #📋 Input data with country names and visit counts
                         x='Country',                          #Countries on X-axis
                         y='VisitCount',                       #📈 Visit count on Y-axis
                         color='VisitCount',                   #🎨 Color bars based on visit frequency
                         color_continuous_scale='Blues',       #🌀 Blue gradient for visual appeal
                         title='Overall Country-wise Visits')  #Chart title
    st.plotly_chart(fig_country, use_container_width=True)     #Renders the interactive Plotly bar chart in Streamlit with full width

    #Continent-wise Overall
    continent_counts = df_tx_full['Continent'].value_counts().reset_index()
    #📊 Count the number of visits for each continent and convert it to a DataFrame

    continent_counts.columns = ['Continent', 'VisitCount']
    #📝 Rename the columns for clarity (from default ['index', 'Continent'] to readable names)

    fig_cont = px.pie(continent_counts,      #📋 Input data containing continent names and visit counts
                      names='Continent',     #🏷️ Labels shown on the pie chart
                      values='VisitCount',   #📈 Slice size based on number of visits
                      title='Overall Continent-wise Visits') #Chart title
    st.plotly_chart(fig_cont, use_container_width=True) # Display the pie chart in Streamlit with full container width for better layout


    # 🗓️ YEAR-WISE BREAKDOWN
    st.subheader("📅 Year-wise Visit Breakdown")
    #Adds a subheader to visually separate this section in the Streamlit app
    if 'VisitYear' in df_tx_full.columns:
        #Check if the VisitYear column is available before proceeding to avoid errors

        selected_year = st.selectbox("📅 Select Year:", sorted(df_tx_full['VisitYear'].dropna().unique(), reverse=True))
        #Creates a dropdown with available years in descending order for user to select

        df_year_filtered = df_tx_full[df_tx_full['VisitYear'] == selected_year]
        #Filters the data to include only the rows for the selected year
        
        #Region-wise Visits (Year Filtered)
        region_yr = df_year_filtered['Region'].value_counts().reset_index()
        #📊 Count the visits by region for the selected year and convert to DataFrame

        region_yr.columns = ['Region', 'VisitCount'] #Rename the columns for clarity

        fig_yr_region = px.bar(region_yr, 
                               x='Region', 
                               y='VisitCount', 
                               color='VisitCount',
                               color_continuous_scale='viridis', 
                               title=f'Region-wise Visits in {selected_year}')
        st.plotly_chart(fig_yr_region, use_container_width=True) #📈 Generate a bar chart to visualize region-level visits for the selected year

        #Country-wise Visits (Year Filtered)
        country_yr = df_year_filtered['Country'].value_counts().reset_index() #📊 Count the visits by country for the selected year
        country_yr.columns = ['Country', 'VisitCount']                        #Rename columns for clarity
        fig_yr_country = px.bar(country_yr, 
                                x='Country', 
                                y='VisitCount', 
                                color='VisitCount',
                                color_continuous_scale='Blues', 
                                title=f'Country-wise Visits in {selected_year}') #📈 Create bar chart for country-wise visits in that year
        st.plotly_chart(fig_yr_country, use_container_width=True) #🖼️ Display country bar chart in full width   
        
        #🌎 Continent-wise Visits (Year Filtered)
        cont_yr = df_year_filtered['Continent'].value_counts().reset_index() #📊 Count visits by continent for the selected year
        cont_yr.columns = ['Continent', 'VisitCount']       #📝 Rename columns for clarity
        fig_yr_cont = px.pie(cont_yr, 
                             names='Continent', 
                             values='VisitCount',
                             title=f'Continent-wise Visits in {selected_year}') #🥧 Create a pie chart to visualize visit share by continent
        st.plotly_chart(fig_yr_cont, use_container_width=True) #🖼️ Display the pie chart with full width

        # 📂 Download Option
        download_df = df_year_filtered[['Country', 'Region', 'Continent', 'VisitYear']]  #Create a filtered DataFrame with selected columns for export
        download_format = st.radio("📄 Download Year-wise Summary As:", 
                                   ['CSV', 'Excel'], 
                                   horizontal=True) 
        #Add a horizontal radio button for users to choose between CSV or Excel formats
        
        #CSV Download Option
        if download_format == 'CSV':
            st.download_button("⬇️ Download CSV", 
                               download_df.to_csv(index=False).encode('utf-8'),
                               f"Year_{selected_year}_Visits.csv", 
                               mime="text/csv")
        #💾 Converts DataFrame to CSV and allows user to download it

        #Excel Download Option   
        else:
            output = io.BytesIO() #📦 Create a memory buffer to temporarily hold Excel file in memory
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                download_df.to_excel(writer, index=False, sheet_name=f'Visits_{selected_year}')
        #📊 Write the DataFrame to an Excel sheet in memory using XlsxWriter   
            
            st.download_button("⬇️ Download Excel", 
                               output.getvalue(), 
                               f"Year_{selected_year}_Visits.xlsx",
                               mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        #💾 Allow download of the Excel file with correct MIME type    

    #Short Description:
    #This page provides region-level travel insights based on actual user visits
    #Users can explore both overall and year-wise travel patterns across continents, countries, and regions

    #🔄 PAGE TRIGGER:
    # elif page == "🌍 Get Recommendations"  → Executes this logic when selected from the sidebar

    # 🎯 Purpose:
    # To deliver geo-based tourism analytics by:
    # - Merging transactional data with region, country, and continent mappings
    # - Visualizing visit patterns using interactive charts
    # - Enabling year-wise filtering and downloadable visit summaries for deeper insights

    #🌐 USE CASES:
    # - Understand which locations are most frequently visited
    # - Analyze visit trends over years at geographic levels
    # - Build location-based filters for recommendation or visualization

    # 🔄 Workflow Breakdown:
    # 1. Data Loading & Cleaning:
    #    - Loaded multiple sheets (User, Transaction, Country, Region, Continent)
    #    - Cleaned column names and filled missing values
    
    # 2. Overall Visit Analysis:
    #    - Merged user-transaction with regional hierarchy
    #    - Displayed total visit counts by region, country, and continent
    
    # 3. Year-wise Visit Breakdown:
    #    - Enabled filtering by selected year
    #    - Plotted visit counts by geography for that specific year
    
    # 4. Download Feature:
    #    - Export visit data by year in CSV or Excel format

    # ✨ KEY FEATURES IMPLEMENTED:
    # - Interactive region/country/continent analysis
    # - Year selection for historical trend exploration
    # - Visual bar and pie charts using Plotly
    # - Exportable visit data summaries

    # 🤖 MODELS & TECHNIQUES USED:
    # - Exploratory Data Analysis (EDA) using aggregation methods like `groupby()` and `value_counts()`
    # - No machine learning models applied — focused on descriptive visual analytics

    #TOOLS & LIBRARIES USED:
    # Streamlit
    # - st.selectbox(), st.radio(), st.download_button() → For UI and interaction
    # - st.subheader(), st.markdown()                    → Section layout and explanations
    # - st.plotly_chart()                                → For displaying insights

    # Pandas
    # - pd.read_excel(), merge(), groupby(), value_counts() → Data transformation

    # Plotly Express
    # - px.bar(), px.pie() → For bar and pie charts

    #Commands
    # 📘 STREAMLIT
    # - st.title()                   → Adds a main title to the page
    # - st.markdown()                → Displays rich text/HTML-style descriptions
    # - st.subheader()               → Adds smaller section headers within the page
    # - st.selectbox()               → Creates a dropdown selector for year selection
    # - st.radio()                   → Lets users choose between CSV and Excel format
    # - st.download_button()         → Adds a button to download processed data
    # - st.plotly_chart()            → Displays interactive Plotly charts in the app
    # - @st.cache_data()             → Caches data-loading functions to avoid re-reading the file

    # 📊 PANDAS
    # - pd.read_excel()              → Reads data from individual Excel sheets
    # - df.columns.str.strip()       → Strips whitespace from column names
    # - df.fillna("Missing")         → Fills missing string values with "Missing"
    # - df.apply(lambda col: ...)    → Applies column-wise transformation (e.g., handling nulls)
    # - df.merge()                   → Combines DataFrames on common keys
    # - df.value_counts()            → Counts unique occurrences of values in a column
    # - df.groupby()                 → Used for aggregation (if added later in enhancements)
    # - df.to_csv()                  → Converts DataFrame to CSV format for download
    # - pd.ExcelWriter()             → Used to write DataFrame into Excel with formatting

    # 📈 PLOTLY EXPRESS
    # - px.bar()                     → Creates interactive bar charts
    # - px.pie()                     → Creates interactive pie charts


    #🧭 PAGE SUMMARY:
    # - Goal               : Explore region, country, and continent-level user visit data  
    # - Datasets Used      : User, Transaction, Country, Region, Continent  
    # - Features Used      : Location fields (RegionId, CountryId, ContinentId), VisitYear  
    # - Output             : Cleaned and structured data used to build filters, insights, charts  
    # - Model Used         : None (this page is based on descriptive filtering, not ML models) 

    # 📊 OUTPUT SECTIONS:
    # 1. Overall Region-wise Visits
    # 2. Overall Country-wise Visits
    # 3. Overall Continent-wise Visits
    # 4. Year-wise Filtered Insights (Region, Country, Continent)
    # 5. Downloadable Summary Table


#Overall Summary 

#| Page                        | Type            | Unique Feature                          |
#| --------------------------  | --------------- | ----------------------------------------|
#| 🏠 Home                    | Intro           | Greeting, app overview                   |
#| 🔍 Data Explorer           | Raw Data Viewer | View & export any sheet                  |
#| 📊 User Summary Statistics | Recommendation  | Personalized suggestions + user segments |
#| 📈 Analyze Trends          | EDA / Trends    | Year-wise trends by region/type/mode     |
#| 🧮 Predict Visit Mode      | Classification  | Predict purpose (Family, Business)       |
#| 📈 Predict Ratings         | Regression      | Predict rating scores for attractions    |
#| 🌍 Get Recommendations     | Recommendation  | Real-time top-N attraction suggestions   |
    

#🏠 Home
#This page gives a welcome message and explains what the app does
#Just an introduction

#🔍 Data Explorer
#This page lets you view and download raw data tables (like User, Transaction, etc.)
#You can clean, filter, and plot basic charts here

#📊 User Summary Statistics
#This page gives personalized attraction suggestions for a selected user
#Also shows user segments and trends based on behavior

#📈 Analyze Trends
#This page shows visit trends by year, region, country, visit mode, etc
#Can choose a year and see charts about tourism patterns

#🧮 Predict Visit Mode
#This page predicts the reason for a trip (Family, Business, etc.) using a machine learning model
#Can see the accuracy and prediction results

#📈 Predict Ratings
#This section predicts the rating (1–5) a user might give to a tourist attraction
    #using a regression model trained on visit details and user demographics.
#It also displays model performance metrics such as R², MSE, and RMSE.

#🌍 Get Recommendations
#This page suggests new attractions for a user based on their history or similar users
#Uses collaborative and content-based recommendation logic
