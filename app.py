
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.metrics import average_precision_score, auc, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

raw_X = pd.read_csv('Spotifiy Streamlit Data')
raw_X.head(1)
X = raw_X.copy()
X.drop('Charted', inplace=True, axis=1)
X.head(1)

raw_X.describe()

# Change year to a categorical variable
X.year = X.year.astype('O')
X.info()

X.describe()

X.Main_artist.unique()

X.Featured_artist.unique()

st.title('Spotify Hit Song Predictor')

st.sidebar.header('Specify Input Parameters')

acousticness = st.sidebar.slider('Acousticness', 0.0, 1.0, 0.5)
danceability = st.sidebar.slider('Danceability', 0.0, 1.0, 0.5)
duration_ms = (st.sidebar.slider('Duration in minutes', 1.0, 10.0, value=3.0, step=0.5))*60000
energy = st.sidebar.slider('Energy', 0.0, 1.0, 0.5)
explicit = st.sidebar.slider('Explicit', 0, 1, step=1)
instrumentalness = st.sidebar.slider('Instrumentalness', 0.0, 1.0, 0.5)
key = st.sidebar.slider('Key', 0, 11, step=1)
liveness = st.sidebar.slider('Liveness', 0.0, 1.0, 0.5)
loudness = (st.sidebar.slider('Loudness in db', 0, 10, step=1, value=6)) * -1
mode = st.sidebar.slider('Mode', 0.0, 1.0, 0.5)
popularity = st.sidebar.slider('Popularity', 0, 100, step=1, value=50)
speechiness = st.sidebar.slider('Speechiness', 0.0, 1.0, 0.2)
tempo = st.sidebar.slider('Tempo', 0, 230, 120, step=1)
valence = st.sidebar.slider('Valence', 0.0, 1.0, 0.5)
year = st.sidebar.selectbox('Year', ('2017', '2018', '2019', '2020'))
Main_artist = st.sidebar.selectbox('Artist', ('Other', 'XXXTENTACION', 'Lil Uzi Vert', 'Ed Sheeran', 'Khalid',
                                              '21 Savage', 'Billie Eilish', 'Drake', 'Future', 'Miley Cyrus',
                                              'Playboi Carti', 'Trippie Redd', 'Lil Baby', 'Juice WRLD',
                                              'Travis Scott', 'Post Malone', 'Mac Miller', 'Cardi B',
                                              'Ariana Grande', 'DaBaby', 'J. Cole', 'Pop Smoke', 'Dua Lipa',
                                              'Bad Bunny', 'The Kid LAROI', 'Taylor Swift', 'Kid Cudi', 'BTS',
                                              'Eminem', 'Ozuna'))
Featured_artist = st.sidebar.selectbox('Featured artist', ('No features', 'Other', 'Travis Scott', 'Metro Boomin', 'Swae Lee',
                                                           'Lil Uzi Vert', 'Gunna', 'Drake', 'Juice WRLD', 'Bad Bunny',
                                                           'Young Thug', 'YoungBoy Never Broke Again', 'Roddy Ricch',
                                                           'Justin Bieber', 'Nicki Minaj', 'Lil Baby', 'DaBaby', 'Marshmello',
                                                           'Post Malone', 'Halsey', 'Daddy Yankee', 'J Balvin'))


data = {'acousticness': acousticness,
        'danceability': danceability,
        'duration_ms': duration_ms,
        'energy': energy,
        'explicit': explicit,
        'instrumentalness': instrumentalness,
        'key': key,
        'liveness': liveness,
        'loudness': loudness,
        'mode': mode,
        'popularity': popularity,
        'speechiness': speechiness,
        'tempo': tempo,
        'valence': valence,
        'year': year,
        'Main_artist': Main_artist,
        'Featured_artist': Featured_artist}


features = pd.DataFrame(data, index=[0])


df = pd.concat([features, X], axis=0)


df = pd.get_dummies(df)
#df.drop('year_2017', inplace=True, axis=1)

df = df.loc[:, ~df.columns.duplicated()]
df2 = df.copy()
scaler = StandardScaler()
df = scaler.fit_transform(df)

df = df[:1]  # Selects only the first row (the user input data)


# Displays the user input features
st.subheader('User Input features')
st.write(df)


# Reads in saved classification model
load_clf = pickle.load(open('spotify_model.pkl', 'rb'))
load_clf
# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)[:, 1]
prediction_proba = int(prediction_proba * 100)

if prediction == 1:
    st.subheader('This song has a {}% likelihood of becoming a hit song!'.format(prediction_proba))
else:
    st.subheader('Unfortunately this song has a {}% likelihood of not becoming a hit song.'.format(
        100-prediction_proba))

feature_importance = pd.DataFrame(
    pd.Series(load_clf.feature_importances_, df2.columns).nlargest(10))
feature_importance = feature_importance.rename(columns={0: 'Most important features'})
feature_importance = feature_importance.sort_values(by='Most important features')
st.subheader('Top 10 attributes for a hit song')
fig_bar = st.bar_chart(feature_importance)
