import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
import os

meta=pd.read_csv("C:\\Users\\snair\\Documents\\Data Science Assignment\\New Projects\\Recommender\\Movie_Recommendation_System\\datasets\\movie_metadata.csv")
met=pd.read_csv("C:\\Users\\snair\\Documents\\Data Science Assignment\\New Projects\\Recommender\\Movie_Recommendation_System\\datasets\\movies_metadata.csv")

meta.columns
data=meta.loc[:,["movie_title","director_name","duration","actor_1_name","actor_2_name","actor_3_name","gross","genres","plot_keywords","budget","title_year"]]
data.rename(columns={"movie_title":"Title","director_name":"director","actor_1_name":"actor_1","actor_2_name":"actor_2","actor_3_name":"actor_3","title_year":"year"},inplace=True)

data.describe()
data.loc[data.duplicated()]
data.drop_duplicates(inplace=True)
data=data.reset_index(drop=True)
data.isna().sum()


data["Title"]=data.Title.apply(lambda x:x.split('\xa0')[0])

met.columns
met.describe()
met.release_date.fillna(0,inplace=True)
met.isna().sum()

met.loc[met.release_date=='1']
met.loc[met.release_date=='12']
met.loc[met.release_date=='22']
met.drop([19730],inplace=True)
met.drop([29503],inplace=True)
met.drop([35587],inplace=True)

met["year"]=met["release_date"].apply(lambda x: datetime.strptime(x,"%Y-%m-%d").year if x!=0 else 0)


def impute_from_met(col_data,col_met):
    a=data.loc[data[col_data].isna(),"Title"].reset_index()
    b=met.title.isin(list(a.Title))
    c=met[b][[col_met,"title"]].reset_index()
    c.rename(columns={"index":"met_index","title":"Title"},inplace=True)
    
    d=a.merge(c,on="Title",how='left')
    x=d.loc[d.Title.duplicated()].index
    d.drop(x,axis=0,inplace=True)
    d=d.reset_index(drop=True)
    
    for i in range(len(d)):
        data.loc[d["index"][i],col_data]=d[col_met][i]
    

impute_from_met("year","year")
impute_from_met("duration","runtime")
impute_from_met("gross","revenue")
impute_from_met("budget","budget")

data.isna().sum()


data["year"].value_counts().sort_index()#movies till 2016 are present

data["budget"]=data["budget"].astype(float)

data["genres"]=data["genres"].str.replace('|',' ')
data["plot_keywords"]=data["plot_keywords"].str.replace('|',' ')

sns.distplot(data["year"].dropna())
sns.distplot(data["duration"].dropna())
sns.distplot(data["gross"].dropna())
sns.distplot(data["budget"].dropna()) 

#####################################################################################

met.loc[met["year"]>2016]
met["year"].value_counts().sort_index()
#data of 2017 can be fetched from this

met.columns
movies_2017=met.loc[met.year==2017,["id","title","runtime","revenue","genres","budget","year"]]
movies_2017=movies_2017.reset_index(drop=True)

import ast  #abstract syntax trees
ast.literal_eval(movies_2017["genres"][3]) #takes the python literals from the string

movies_2017["genres"]=movies_2017["genres"].map(lambda x:ast.literal_eval(x))


def extract_genres(feature):
    genre=[]
    for i in range(len(movies_2017)):
        for j in range(len(movies_2017[feature][i])):
            genre.append(movies_2017[feature][i][j].get('name'))
        movies_2017[feature][i]=' '.join(genre)
        genre=[]


extract_genres("genres")


credit=pd.read_csv("C:\\Users\\snair\\Documents\\Data Science Assignment\\New Projects\\Recommender\\Movie_Recommendation_System\\datasets\\credits.csv")

movies_2017["id"]=movies_2017["id"].astype(int)
movies_2017=movies_2017.merge(credit,on='id')
movies_2017["cast"]=movies_2017["cast"].map(lambda x:ast.literal_eval(x))
movies_2017["crew"]=movies_2017["crew"].map(lambda x:ast.literal_eval(x))


def extract_director(feature):
    movies_2017["director"]=""
    for i in range(len(movies_2017)):
        for j in range(len(movies_2017[feature][i])):
            if movies_2017[feature][i][j].get('department')=="Directing":
                movies_2017["director"][i]=movies_2017[feature][i][j].get('name')

def extract_actors(feature):
    movies_2017["actor_1"]=""
    movies_2017["actor_2"]=""
    movies_2017["actor_3"]=""
    for i in range(len(movies_2017)):
        try:
            movies_2017["actor_1"][i]=movies_2017[feature][i][0].get('name')
            movies_2017["actor_2"][i]=movies_2017[feature][i][1].get('name')
            movies_2017["actor_3"][i]=movies_2017[feature][i][2].get('name')
        except:
            pass
 
extract_director("crew")               
extract_actors("cast")           
        

movies_2017.drop(["cast","crew"],axis=1,inplace=True)
        
movies_2017.isna().sum()
movies_2017=movies_2017.fillna(0)        
movies_2017["budget"]=movies_2017["budget"].astype(float)    
    
movies_2017.columns
movies_2017.loc[(movies_2017.genres=='') & (movies_2017.director=='') & (movies_2017.actor_1=='')]       
movies_2017.drop([370],inplace=True) # dropping row with no genre,director or actors       

data.columns
movies_2017["plot_keywords"]=""
movies_2017=movies_2017[['title','director','runtime','actor_1','actor_2','actor_3','revenue','genres','plot_keywords','budget','year']]        
movies_2017=movies_2017.rename(columns={"title":"Title","runtime":"duration","revenue":"gross"})   
 
data=data.append(movies_2017,ignore_index=True)       

os.chdir("C:/Users/snair/Documents/Data Science Assignment/New Projects/Recommender/Movie_Recommendation_System/datasets")
data.to_csv("data_till_2017",encoding='utf-8')

###################################################################################        
        
data=pd.read_csv("C:/Users/snair/Documents/Data Science Assignment/New Projects/Recommender/Movie_Recommendation_System/datasets/data_till_2017.csv")
data.isnull().sum()

#2018 movies from https://en.wikipedia.org/wiki/List_of_American_films_of_2018
url_2018="https://en.wikipedia.org/wiki/List_of_American_films_of_2018"

movies_2018=pd.read_html(url_2018)
movies_2018=movies_2018[2:6]
movies_18=movies_2018[0].append(movies_2018[1].append(movies_2018[2].append(movies_2018[3])))

movies_18.columns
movies_18=movies_18[["Title","Cast and crew"]].reset_index(drop=True)    


def get_director(x):
    if (" (director" in x):
        return x.split(" (director")[0]


def get_actor(x,i):
    try:
        if "(edited); " in x:
            return x.split("(edited); ")[1].split(', ')[i]
        elif "(screenplay); " in x:
            return x.split("(screenplay); ")[1].split(', ')[i]
        elif "screenplay); " in x:
            return x.split("screenplay); ")[1].split(', ')[i]
        elif "(screenplay), " in x:
            return x.split("(screenplay), ")[1].split(', ')[i]
        elif "(Screenplay), " in x:
            return x.split("(Screenplay), ")[1].split(', ')[i]
        elif "(writer); " in x:
            return x.split("(writer); ")[1].split(', ')[i]
        elif "(director); " in x:
            return x.split("(director); ")[1].split(', ')[i]
    except:
        pass


movies_18["director"]=movies_18["Cast and crew"].map(lambda x: get_director(x))
movies_18["actor_1"]=movies_18["Cast and crew"].map(lambda x: get_actor(x,0))
movies_18["actor_2"]=movies_18["Cast and crew"].map(lambda x: get_actor(x,1))
movies_18["actor_3"]=movies_18["Cast and crew"].map(lambda x: get_actor(x,2))
movies_18["year"]=2018
movies_18.drop(["Cast and crew"],axis=1,inplace=True)

#movies_18["Cast and crew"][153]
#movies_18["Cast and crew"][194]
#movies_18["Cast and crew"][235]
#movies_18["Cast and crew"][236]
#movies_18["Cast and crew"][248]

movies_18.isna().sum()

#Pulling additional data from TMDB using api

from tmdbv3api import TMDb,Movie
import requests

tmdb=TMDb()
tmdb.api_key="ca6fe346758a39608960f36d07fb3a75"
movie=Movie()

def get_genres(x):
    try:
        movie_id=movie.search(x)[0].id
    except:
        movie_id=movie.search(x.lower())[0].id
    finally:
        response=requests.get('https://api.themoviedb.org/3/movie/{}?api_key={}'.format(movie_id,tmdb.api_key))
        response=response.json()
        genre=[]
        for i in range(len(response.get('genres'))):
            genre.append(response.get('genres')[i].get('name'))            
        return " ".join(genre)
    

def get_duration(x):
    try:
        movie_id=movie.search(x)[0].id
    except:
        movie_id=movie.search(x.lower())[0].id
    finally:
        response=requests.get('https://api.themoviedb.org/3/movie/{}?api_key={}'.format(movie_id,tmdb.api_key))
        response=response.json()
        return response.get('runtime')


def get_gross(x):
    try:
        movie_id=movie.search(x)[0].id
    except:
        movie_id=movie.search(x.lower())[0].id
    finally:
        response=requests.get('https://api.themoviedb.org/3/movie/{}?api_key={}'.format(movie_id,tmdb.api_key))
        response=response.json()
        return response.get('revenue')


def get_budget(x):
    try:
        movie_id=movie.search(x)[0].id
    except:
        movie_id=movie.search(x.lower())[0].id
    finally:
        response=requests.get('https://api.themoviedb.org/3/movie/{}?api_key={}'.format(movie_id,tmdb.api_key))
        response=response.json()
        return response.get('budget')


movies_18["genres"]=movies_18["Title"].map(lambda x: get_genres(x))
movies_18["duration"]=movies_18["Title"].map(lambda x: get_duration(x))
movies_18["gross"]=movies_18["Title"].map(lambda x: get_gross(x))
movies_18["budget"]=movies_18["Title"].map(lambda x: get_budget(x))
movies_18["plot_keywords"]=""
movies_18.columns
movies_18=movies_18[["Title","director","duration","actor_1","actor_2","actor_3","gross","genres","plot_keywords","budget","year"]]

data=data.append(movies_18,ignore_index=True)

os.chdir("C:/Users/snair/Documents/Data Science Assignment/New Projects/Recommender/Movie_Recommendation_System/datasets")
data.to_csv("data_till_2018",encoding='utf-8')

#####################################################################################

data=pd.read_csv("C:/Users/snair/Documents/Data Science Assignment/New Projects/Recommender/Movie_Recommendation_System/datasets/data_till_2018.csv")
data.isnull().sum()

url_2019="https://en.wikipedia.org/wiki/List_of_American_films_of_2019"
movies_2019=pd.read_html(url_2019)
movies_2019=movies_2019[3:7]

movies_19=movies_2019[0].append(movies_2019[1].append(movies_2019[2].append(movies_2019[3])))

movies_19.columns
movies_19=movies_19[["Title","Cast and crew"]].reset_index(drop=True)    

movies_19["director"]=movies_19["Cast and crew"].map(lambda x: get_director(x))
movies_19["actor_1"]=movies_19["Cast and crew"].map(lambda x: get_actor(x,0))
movies_19["actor_2"]=movies_19["Cast and crew"].map(lambda x: get_actor(x,1))
movies_19["actor_3"]=movies_19["Cast and crew"].map(lambda x: get_actor(x,2))
movies_19["year"]=2019
movies_19.drop(["Cast and crew"],axis=1,inplace=True)

movies_19["genres"]=movies_19["Title"].map(lambda x: get_genres(x))
movies_19["duration"]=movies_19["Title"].map(lambda x: get_duration(x))
movies_19["gross"]=movies_19["Title"].map(lambda x: get_gross(x))
movies_19["budget"]=movies_19["Title"].map(lambda x: get_budget(x))
movies_19["plot_keywords"]=""
movies_19.columns
movies_19=movies_19[["Title","director","duration","actor_1","actor_2","actor_3","gross","genres","plot_keywords","budget","year"]]

data=data.append(movies_19,ignore_index=True)

###############################################################################

url_2020="https://en.wikipedia.org/wiki/List_of_American_films_of_2020"
movies_2020=pd.read_html(url_2020)
movies_2020=movies_2020[3:7]

movies_20=movies_2020[0].append(movies_2020[1].append(movies_2020[2].append(movies_2020[3])))

movies_20.columns
movies_20=movies_20[["Title","Cast and crew"]].reset_index(drop=True)    

movies_20["director"]=movies_20["Cast and crew"].map(lambda x: get_director(x))
movies_20["actor_1"]=movies_20["Cast and crew"].map(lambda x: get_actor(x,0))
movies_20["actor_2"]=movies_20["Cast and crew"].map(lambda x: get_actor(x,1))
movies_20["actor_3"]=movies_20["Cast and crew"].map(lambda x: get_actor(x,2))
movies_20["year"]=2020
movies_20.drop(["Cast and crew"],axis=1,inplace=True)

movies_20["genres"]=movies_20["Title"].map(lambda x: get_genres(x))
movies_20["duration"]=movies_20["Title"].map(lambda x: get_duration(x))
movies_20["gross"]=movies_20["Title"].map(lambda x: get_gross(x))
movies_20["budget"]=movies_20["Title"].map(lambda x: get_budget(x))
movies_20["plot_keywords"]=""
movies_20.columns
movies_20=movies_20[["Title","director","duration","actor_1","actor_2","actor_3","gross","genres","plot_keywords","budget","year"]]

data=data.append(movies_20,ignore_index=True)

os.chdir("C:/Users/snair/Documents/Data Science Assignment/New Projects/Recommender/Movie_Recommendation_System/datasets")
data.to_csv("data_till_2020",encoding='utf-8')

#########################################################################################
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

data=pd.read_csv("C:/Users/snair/Documents/Data Science Assignment/New Projects/Recommender/Movie_Recommendation_System/datasets/data_till_2020.csv")
data.isnull().sum()
data[data.duplicated()]

data["duration"]=data["duration"].replace(np.nan,0)
data["gross"]=data["gross"].replace(np.nan,0)
data["budget"]=data["budget"].replace(np.nan,0)
data["year"]=data["year"].replace(np.nan,0)
data["director"]=data["director"].replace(np.nan,"")
data["actor_1"]=data["actor_1"].replace(np.nan,"")
data["actor_2"]=data["actor_2"].replace(np.nan,"")
data["actor_3"]=data["actor_3"].replace(np.nan,"")
data["genres"]=data["genres"].replace(np.nan,"")
data["plot_keywords"]=data["plot_keywords"].replace(np.nan,"")

data.isnull().sum()                
        
data["genres"]=data["genres"].str.replace("Science Fiction","Sci-Fi")

lem=WordNetLemmatizer()
stop=list(stopwords.words('english'))  
  
def lemmatizer(x):
    words=re.sub('[^A-Za-z0-9' ']+',' ',x)
    words=nltk.word_tokenize(words)
    words=[lem.lemmatize(word) for word in words if word not in stop]
    words = list(dict.fromkeys(words))
    return " ".join(words)

data["keywords"]=data["plot_keywords"].map(lambda x: lemmatizer(x))

data["combination"]=data["director"].str.replace(' ','')+' '+data["actor_1"].str.replace(' ','')+' '+data["actor_2"].str.replace(' ','')+' '+data["actor_3"].str.replace(' ','')+' '+data["genres"]+' '+data["keywords"]

os.chdir("C:/Users/snair/Documents/Data Science Assignment/New Projects/Recommender/Movie_Recommendation_System/datasets")
data.to_csv("data_final",encoding='utf-8')

######################################################################################
data=pd.read_csv("C:/Users/snair/Documents/Data Science Assignment/New Projects/Recommender/Movie_Recommendation_System/datasets/data_final.csv")


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

cv=CountVectorizer()
count_matrix=cv.fit_transform(data["combination"])
similarity = cosine_similarity(count_matrix)

import pickle
os.chdir("C:\\Users\\snair\\Documents\\Data Science Assignment\\New Projects\\Recommender\\Movie_Recommendation_System")
pickle.dump(similarity,open('cos_matrix.pkl','wb'))


movie="Captain America: The Winter Soldier"
index=data.loc[data["Title"]==movie].index[0]
similarity_scores=list(enumerate(similarity[index]))
similarity_scores=sorted(similarity_scores,key=lambda x: x[1],reverse=True)

top_scores=[i[1] for i in similarity_scores[1:11]]
top_index=[i[0] for i in similarity_scores[1:11]]

data.loc[top_index,"Title"]



#from tmdbv3api import TMDb,Movie
#import requests
#
#tmdb=TMDb()
#tmdb.api_key="ca6fe346758a39608960f36d07fb3a75"
#mov=Movie()
#movie_id=mov.search(movie)[0].id
#
#response=requests.get('https://api.themoviedb.org/3/movie/{}?api_key={}'.format(movie_id,tmdb.api_key))
#response=response.json()

#data.loc[data.Title=="Baahubali 2: The Conclusion","Title"]="Bahubali 2: The Conclusion"







