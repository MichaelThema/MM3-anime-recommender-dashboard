import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Dataset, Reader, SVD, CoClustering, BaselineOnly, accuracy
from surprise.model_selection import train_test_split
import mlflow

def load_models_and_data():
    final_model_path = 'final_model.zip'
    tfidf_vectorizer_path = 'tfidf_vectorizer.pkl'
    anime_df_path = 'anime_df.csv'
    train_df_path = 'train_df.zip'

       #Load final_model.pkl from a zip archive
    with zipfile.ZipFile(final_model_path, 'r') as z:
        with z.open('final_model.pkl') as file:
            final_model = joblib.load(file)
    
    #Load tfidf_vectorizer.pkl directly
    with open(tfidf_vectorizer_path, 'rb') as file:
        tfidf_vectorizer = joblib.load(file)
    
    #Load train_df.csv from a zip archive
    with zipfile.ZipFile(train_df_path, 'r') as z:
        with z.open('train_df.csv') as file:
            train_df = pd.read_csv(file)
    
    # Load anime_df.csv directly
    anime_df = pd.read_csv(anime_df_path)
    
    return final_model, tfidf_vectorizer, anime_df, train_df
def plot_correlation_matrix(df):
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)

def plot_rating_distribution(train_df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(train_df['rating'], bins=10, kde=True, ax=ax)
    ax.set_title("Distribution of Ratings in train_df")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

def plot_ratings_boxplot(train_df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x=train_df['rating'], ax=ax)
    ax.set_title("Box Plot of Ratings in train_df")
    ax.set_xlabel("Rating")
    plt.tight_layout()
    st.pyplot(fig)

def plot_anime_types_count(anime_df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(y=anime_df['type'], order=anime_df['type'].value_counts().index, ax=ax)
    ax.set_title("Count Plot of Anime Types")
    ax.set_xlabel("Count")
    ax.set_ylabel("Type")
    st.pyplot(fig)

def plot_top_genres_count(anime_df):
    fig, ax = plt.subplots(figsize=(8, 6))
    top_genres = anime_df['genre'].value_counts().nlargest(10).index
    sns.countplot(y=anime_df[anime_df['genre'].isin(top_genres)]['genre'], order=top_genres, ax=ax)
    ax.set_title("Count Plot of Top 10 Anime Genres")
    ax.set_xlabel("Count")
    ax.set_ylabel("Genre")
    st.pyplot(fig)

def train_evaluate_models(trainset, testset, algorithm):
    models = {
        'SVD': SVD(),
        'CoClustering': CoClustering(),
        'BaselineOnly': BaselineOnly()
    }

    model = models[algorithm]
    with mlflow.start_run(nested=True):
        model.fit(trainset)
        predictions = model.test(testset)
        rmse = accuracy.rmse(predictions)
        mae = accuracy.mae(predictions)
        st.write(f'{algorithm} - RMSE: {rmse}, MAE: {mae}')
        mlflow.log_param("model", algorithm)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
    return model

def get_recommendations(model, user_id, anime_df, anime_names):
    anime_ids = []
    for anime_name in anime_names:
        anime_id = anime_df[anime_df['name'] == anime_name]['anime_id'].values
        if len(anime_id) > 0:
            anime_ids.append(anime_id[0])
    
    predictions = [model.predict(user_id, anime_id) for anime_id in anime_ids]
    return predictions

def predict_ratings(algorithm, user_id, anime_names, anime_df, train_df):
    # Load and split the data
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(train_df[['user_id', 'anime_id', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.25)

    # Train the selected model
    model = train_evaluate_models(trainset, testset, algorithm)

    # Get recommendations for the given anime names
    predictions = get_recommendations(model, user_id, anime_df, anime_names)
    
    # Format and return predictions
    results = []
    for prediction in predictions:
        anime_title = anime_df[anime_df['anime_id'] == prediction.iid]['name'].values[0]
        results.append(f"Anime: {anime_title}, Estimated Rating: {prediction.est}")
    
    return results

def main():
    st.image("anime.jpeg")

    final_model, tfidf_vectorizer, anime_df, train_df = load_models_and_data()

    st.title('Anime Recommendation System')

    options = ["Information", "Predict Ratings", "Exploratory Data Analysis", "About Us"]
    selection = st.sidebar.selectbox("Choose Option", options)

    if selection == "Information":
        st.info("General Information")
        st.markdown("""Discover Your Next Favorite Anime with Our Smart Recommender System!
        Dive into a world where your anime watching experience is transformed with personalized recommendations! Our cutting-edge recommender system is designed to predict how much you'll love a new anime title based on your unique viewing history and preferences.

        Why Choose Our System?
        What Makes Our System Special?

        Dual Filtering Techniques: 
        We harness the power of both collaborative filtering 
        and content-based filtering to offer you the most 
        accurate predictions. Whether you're a fan of action-packed
        adventures or heartfelt dramas, our system gets you.
        
        Tailored for You:
        By analyzing your historical preferences and the nuances 
        of anime content, we craft recommendations that are as 
        unique as your taste.
        
        End-to-End Development: 
        From data preprocessing and model training to real-time 
        recommendations, we've covered every stage of machine 
        learning development to bring you a seamless experience.
        
        Interactive Streamlit App:
        Explore, interact, and discover with our user-friendly 
        Streamlit application, designed to make finding your 
        next favorite anime fun and easy.
        
        Objective of Our Recommender System:
        Our goal is simple – to enhance your anime watching 
        journey by predicting how you'll rate new titles based 
        on your viewing history. We analyze your preferences and 
        the intricate details of anime content to offer 
        recommendations that truly resonate with you.
        Experience the future of anime recommendations today. 
        Your next favorite anime is just a recommendation away! """)
            
    elif selection == "Exploratory Data Analysis":
        st.subheader("Exploratory Data Analysis")
        analysis_types = st.multiselect("Select analysis types", 
                                        ["Correlation Matrix Heatmap", 
                                        "Distribution of Ratings", 
                                        "Box Plot of Ratings", 
                                        "Count of Different Anime Types", 
                                        "Count of Top 10 Anime Genres"])

        if "Correlation Matrix Heatmap" in analysis_types:
            plot_correlation_matrix(anime_df)
        if "Distribution of Ratings" in analysis_types:
            plot_rating_distribution(train_df)
        if "Box Plot of Ratings" in analysis_types:
            plot_ratings_boxplot(train_df)
        if "Count of Different Anime Types" in analysis_types:
            plot_anime_types_count(anime_df)
        if "Count of Top 10 Anime Genres" in analysis_types:
            plot_top_genres_count(anime_df)

    elif selection == "Predict Ratings":
          st.subheader("Predict Ratings for Anime Titles")
          algorithm = st.radio("Choose algorithm", ["SVD", "CoClustering", "BaselineOnly"])
          user_id = st.number_input("Enter your user ID", min_value=1, step=1)
          anime_name_1 = st.text_input("Enter Anime Name 1", "")
          anime_name_2 = st.text_input("Enter Anime Name 2", "")
          anime_name_3 = st.text_input("Enter Anime Name 3", "")

          if st.button("Predict Ratings"):
              anime_names = [anime_name_1, anime_name_2, anime_name_3]
              anime_names = [name for name in anime_names if name]  # Remove empty names
              if anime_names:
                  results = predict_ratings(algorithm, user_id, anime_names, anime_df, train_df)
                  for result in results:
                      st.write(result)
              else:
                  st.write("Please enter at least one anime name.")
                
    elif selection == "About Us":
        st.info("About Us")
        st.markdown("""For any inquiries or further information, please contact any one from our team: 
                
                            - Mogafe Mogale - nalediandriena@gmail.com
                            - Nolwazi Mndebele - mndebelenf@gmail.com
                            - Nontuthuko Mpanza - nontuthukompanza@outlook.com
                            - Seneme Mpungose - senemempungose16@gmail.com
                            - Michael Thema - michaelthema@gmail.com
                            - Thobile Mvuni - thoyomvuni@gmail.com

                              **Vision:**
         To revolutionize information access by seamlessly categorizing news articles, empowering users with accurate and insightful content classification.
         
         **Our Data Scientists:**
         Our team of dedicated data scientists is committed to pushing the boundaries of technology and innovation. With expertise in machine learning, data analysis, and software development, they work tirelessly to enhance the performance and accuracy of our recommender system. Each member brings a unique set of skills and experiences to the project, ensuring that we deliver top-notch recommendations tailored to your preferences.
         
         **Meet the Team:**
         - **Mogafe Mogale:** Expert in machine learning algorithms and data modeling. Passionate about leveraging data to drive insights.
         - **Nolwazi Mndebele:** Skilled in data preprocessing and exploratory analysis. Focused on creating intuitive data visualizations.
         - **Nontuthuko Mpanza:** Specializes in recommendation systems and algorithm optimization. Dedicated to enhancing recommendation accuracy.
         - **Seneme Mpungose:** Experienced in software development and integration. Ensures seamless functionality of our Streamlit app.
         - **Michael Thema:** Adept at statistical analysis and model evaluation. Focused on rigorous testing and validation of models.
         - **Thobile Mvuni:** Expert in data collection and cleaning. Ensures high-quality data for robust recommendations.

         Together, we strive to make your anime watching experience more enjoyable by offering precise and personalized recommendations based on your unique preferences.""")

if __name__ == "__main__":
    main()
