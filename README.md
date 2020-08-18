# Content Based Movie Recommender System
## Overview
There are various movie streaming websites that are in a constant need for customer satisfaction. Recommending movies which share a simlilar character to the movie already seen 
by a viewer has now been a necessity for the websites. This Content Based Movie Recommender System reads the genres, plot, director and actors of a movie and evaluates the 
similarity between other movies and recommends top 10 movies that has the hightest similarity scores.

## Demo 
Link: [https://movie-recommender-system-api.herokuapp.com](https://movie-recommender-system-api.herokuapp.com/)

![](/static/rec_image.JPG)
 
 ## Key Highlights
 * Data Collected from Kaggle
 * Extra data collected from wikipedia by webscrapping
 * Additional data collected from TMDB website using APIs
 * All important features concatenated to form a single feature
 * Similarity matrix made on the feature using cosine similarity
 * Movie names with top similarity scores captured
 * Movie Posters of the selected movie and the recommended movies pulled from TMDB using API
 * Movie posters along with other details passed on to the HTML file
 * IMDB ID of the selected movie pulled from TMDB using API
 * Reviews of the selected movie pulled from IMDB website using IMDB id
 * Sentiment of the reviews found using SentimentIntensityAnalyzer
 * Reviews along with the sentiments passed on to the HTML file
 * Front End built using HTML, CSS and JavaScript
 * Deployment done using Heroku
 
## Technologies Used
<img src="https://www.python.org/static/community_logos/python-logo-master-v3-TM.png" width=280> <img target="_blank" src="https://flask.palletsprojects.com/en/1.1.x/_images/flask-logo.png" width=180> <img target="_blank" src="https://number1.co.za/wp-content/uploads/2017/10/gunicorn_logo-300x85.png" width=280>  <img src="https://www.w3.org/html/logo/badge/html5-badge-h-solo.png" width=80>

## Credits
<img src="static/Kaggle_logo.png" width=280>
<img src="static/tmdb.jpg" width=280>
<img src="https://ia.media-imdb.com/images/M/MV5BMTk3ODA4Mjc0NF5BMl5BcG5nXkFtZTgwNDc1MzQ2OTE@._V1_.png" width=280>
