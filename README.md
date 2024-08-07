# anime-recommender-dashboard

![](https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white) [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](URL_TO_YOUR_APP)

<div id="s_image" align="center">  
  <img src="anime.jpeg" width="550" height="300" alt=""/>  
</div>


## Table of contents
* [1. Project Overview](#project-description)
* [2. Dataset](#dataset)
* [3. Packages](#packages)
* [4. Environment](#environment)
* [5. MLFlow](#mlflow)
* [6. Streamlit](#streamlit)
* [7. Team Members](#team-members)

## 1. Project Overview <a class="anchor" id="project-description"></a>

Our team has been hired as data science consultants for a news outlet to create classification models using Python and deploy it as a web application with Streamlit. 
The aim is to provide you with a hands-on demonstration of applying machine learning techniques to natural language processing tasks.  This end-to-end project encompasses the entire workflow, including data loading, preprocessing, model training, evaluation, and final deployment. The primary stakeholders for the news classification project for the news outlet could include the editorial team, IT/tech support, management, readers, etc. These groups are interested in improved content categorization, operational efficiency, and enhanced user experience.

Anime Recommender System
Project Overview
In today's technology-driven world, recommender systems are socially and economically critical for ensuring that individuals can make appropriate choices surrounding the content they engage with on a daily basis. One application where this is especially true surrounds movie content recommendations; where intelligent algorithms can help viewers find great titles from tens of thousands of options.

Ever wondered how Netflix, Amazon Prime, Showmax, Disney and the likes somehow know what to recommend to you?

Well, you are about to find out! In this project, we build a collaborative and content-based recommender system for a collection of anime titles, capable of accurately predicting how a user will rate an anime title they have not yet viewed, based on their historical preferences.

Dataset Overview
The dataset consists of thousands of users and thousands of anime titles, gathered from myanimelist.net.

## 2. Dataset <a class="anchor" id="dataset"></a>
The dataset is comprised of news articles that need to be classified into categories based on their content, including `Business`, `Technology`, `Sports`, `Education`, and `Entertainment`. You can find both the `train.csv` and `test.csv` datasets [here](https://github.com/ereshia/2401FTDS_Classification_Project/tree/main/Data/processed).

**Dataset Features:**

Files used:
anime.csv: This file contains information about the anime content, including aspects such as the id, name, genre, type, number of episodes (if applicable), an average rating based on views, and the number of members in the anime 'group'.
training.csv: This file contains user ratings of the anime titles.
test.csv: This file will be used to create the rating predictions and must be submitted for grading.
Purpose
The purpose of this project is to develop a robust and reliable recommendation system for anime using collaborative and content-based filtering methods.

Let me know if you'd like me to make any changes!
## 3. Packages <a class="anchor" id="packages"></a>

To carry out all the objectives for this repo, the following necessary dependencies were loaded:
+ `Pandas 2.2.2` and `Numpy 1.26`
+ `Matplotlib 3.8.4`
 

## 4. Environment <a class="anchor" id="environment"></a>

It's highly recommended to use a virtual environment for your projects, there are many ways to do this; we've outlined one such method below. Make sure to regularly update this section. This way, anyone who clones your repository will know exactly what steps to follow to prepare the necessary environment. The instructions provided here should enable a person to clone your repo and quickly get started.

### Create the new evironment - you only need to do this once

```bash
# create the conda environment
conda create --name <env>
```

### This is how you activate the virtual environment in a terminal and install the project dependencies

```bash
# activate the virtual environment
conda activate <env>
# install the pip package
conda install pip
# install the requirements for this project
pip install -r requirements.txt
```
## 5. MLFlow<a class="anchor" id="mlflow"></a>

MLOps, which stands for Machine Learning Operations, is a practice focused on managing and streamlining the lifecycle of machine learning models. The modern MLOps tool, MLflow is designed to facilitate collaboration on data projects, enabling teams to track experiments, manage models, and streamline deployment processes. For experimentation, testing, and reproducibility of the machine learning models in this project, MLflow was used. MLflow helped track hyperparameter tuning by logging and comparing different model configurations. This allows us to easily identify and select the best-performing model based on the logged metrics.

- Please have a look here and follow the instructions: https://www.mlflow.org/docs/2.7.1/quickstart.html#quickstart

## 6. Streamlit<a class="anchor" id="streamlit"></a>

### What is Streamlit?

[Streamlit](https://www.streamlit.io/)  is a framework that acts as a web server with dynamic visuals, multiple responsive pages, and robust deployment of your models.

In its own words:
> Streamlit ... is the easiest way for data scientists and machine learning engineers to create beautiful, performant apps in only a few hours!  All in pure Python. All for free.

> It’s a simple and powerful app model that lets you build rich UIs incredibly quickly.

[Streamlit](https://www.streamlit.io/)  takes away much of the background work needed in order to get a platform which can deploy your models to clients and end users. Meaning that you get to focus on the important stuff (related to the data), and can largely ignore the rest. This will allow you to become a lot more productive.  

##### Description of files

Public Streamlit Repo with application: https://github.com/PrishaniK/OA1_Classification_Project_Streamlit_Repo
Link to Streamlit application: 
For this repository, we are only concerned with a single file:

| File Name              | Description                       |
| :--------------------- | :--------------------             |
| `app.py`               | Streamlit application definition. |
The Streamlit Repo additionally has the pickled models used in the app, the pickled vectoriser and the test.csv 
## Usage Instructions

### Decompressing the Dataset

The `large_dataset.zip` file contains the dataset needed for the project. To decompress it, use the following command:

```sh
unzip train_df.zip
unzip final_model.pkl


#### 6.1 Running the Streamlit web app on your local machine

As a first step to becoming familiar with our web app's functioning, we recommend setting up a running instance on your own local machine. To do this, follow the steps below by running the given commands within a Git bash (Windows), or terminal (Mac/Linux):

- Ensure that you have the prerequisite Python libraries installed on your local machine:

 ```bash
 pip install -U streamlit numpy pandas scikit-learn
 ```

- Navigate to the base of your repo where your base_app.py is stored, and start the Streamlit app.

 ```bash
 cd MM3-animme-recommender-dashboard-/Streamlit/
 streamlit run app.py
 ```

 If the web server was able to initialise successfully, the following message should be displayed within your bash/terminal session:

```
  You can now view your Streamlit app in your browser.

    Local URL: http://localhost:8501
    Network URL: http://192.168.43.41:8501
```
You should also be automatically directed to the base page of your web app. This should look something like:

<div id="s_image" align="center">  
  <img src="app_image.png" width="550" height="300" alt=""/>  
</div>


Congratulations! You've now officially deployed your first web application!

#### 6.2 Deploying your Streamlit web app

- To deploy your app for all to see, click on `deploy`.
  
- Please note: If it's your first time deploying it will redirect you to set up an account first. Please follow the instructions.

## 7. Team Members<a class="anchor" id="team-members"></a>

| Name                                                                                    |  Email              
|-----------------------------------------------------------------------------------------|--------------------------------             
| [Thobile Mvuni](https://github.com/ThobMvuni)                                           | thoyomvuni@gmail.com
| [Michael Thema](https://github.com/MichaelThema)                                        | michaelthema@gmail.com
| [Nolwazi Mndebele](https://github.com/NolwaziMND)                                       | mndebelenf@gmail.com
| [Nontuthuko Mpanza](https://github.com/Nontu98)                                         | nontuthukompanza@outlook.com
| [Seneme Mpungose](https://github.com/Seeneemee)                                         | senemempungose16@gmail.com
| [mogafe mogale](https://github.com/Andriena)                                            | nalediandriena@gmail.com
