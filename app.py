from flask import Flask,request,render_template
import pandas as pd
import numpy as np
from tmdbv3api import TMDb,Movie
import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app=Flask(__name__)

data=pd.read_csv("C:/Users/snair/Documents/Data Science Assignment/New Projects/Recommender/Movie_Recommendation_System/datasets/data_final.csv")
data["director"]=data["director"].replace(np.nan,"")
data["actor_1"]=data["actor_1"].replace(np.nan,"")
data["actor_2"]=data["actor_2"].replace(np.nan,"")
data["actor_3"]=data["actor_3"].replace(np.nan,"")
data["genres"]=data["genres"].replace(np.nan,"")

#response="Bahubali 2: The Conclusion"
#title="1911"
tmdb=TMDb()
tmdb.api_key="ca6fe346758a39608960f36d07fb3a75"
mov=Movie()
sia=SentimentIntensityAnalyzer()

response=""
def create_cos_matrix():
    cv=CountVectorizer()
    count_matrix=cv.fit_transform(data["combination"])
    cos_matrix = cosine_similarity(count_matrix)
    return cos_matrix

def get_movie_poster(title): 
    movie_id=mov.search(title)[0].id
    tmdb_resp=requests.get('https://api.themoviedb.org/3/movie/{}?api_key={}'.format(movie_id,tmdb.api_key))
    tmdb_resp=tmdb_resp.json()
    poster_path=tmdb_resp.get("poster_path")
    if poster_path:
        return "https://image.tmdb.org/t/p/w500"+poster_path
    else:
        return "https://image.tmdb.org/t/p/w500/6EiRUJpuoeQPghrs3YNktfnqOVh.jpg"
    
def get_other_details(index):
    movie_details={"Poster":"","Title":data.loc[index,"Title"],"Director":data.loc[index,"director"],"Cast":data.loc[index,"actor_1"]+','+data.loc[index,"actor_2"]+','+data.loc[index,"actor_3"],"Genres":data.loc[index,"genres"]}
    return movie_details

def get_sentiment(review):
    analysis=sia.polarity_scores(review)
    if analysis['compound']>0:
        return 'positive'        
    elif analysis['compound']<0:
        return 'negative'        
    else:
        return 'neutral'


@app.route('/reviews')
def reviews():
    global response
    movie_id=mov.search(response)[0].id
    tmdb_resp=requests.get('https://api.themoviedb.org/3/movie/{}?api_key={}'.format(movie_id,tmdb.api_key))
    tmdb_resp=tmdb_resp.json()
    imdb_id=tmdb_resp.get("imdb_id")
    resp=requests.get('https://www.imdb.com/title/{}/reviews?ref_=tt_ov_rt'.format(imdb_id))
    soup=BeautifulSoup(resp.content,'html.parser')
    reviews=soup.find_all(attrs={"class","text show-more__control"})
    review=[]
    for rev in reviews:
        review.append(rev.text)
    review=pd.Series(review)
    sentiment=review.apply(lambda x: get_sentiment(x))
    reviews = dict(zip(review,sentiment))
    return render_template("reviews.html",reviews=reviews) 
    

@app.route('/recommend',methods=['POST'])
def recommend():
    global response
    response=request.form["Movie"]
    try:
        index=data.loc[data.Title==response].index[0]
    except:
        return render_template("index.html",error="Sorry! Movie not in our database")
    cos_matrix=create_cos_matrix()
    similarity_scores=list(enumerate(cos_matrix[index]))
    similarity_scores=sorted(similarity_scores,key=lambda x: x[1],reverse=True)
    top_index=[i[0] for i in similarity_scores[:11]]
    recommend=data.loc[top_index,"Title"]
    
    recommend=pd.DataFrame(recommend).reset_index()
    poster=list(recommend["Title"].map(lambda x: get_movie_poster(x)))
    movie_dict=list(recommend["index"].map(lambda x: get_other_details(x)))
    
    for i in range(11): 
        movie_dict[i].update({"Poster":poster[i]})

    return render_template("recommend.html",recommended=movie_dict) 


@app.route('/')
def home():
    return render_template("index.html")

if __name__=="__main__":
    app.run(debug=True)

data.loc[data.Title=="1911"]
