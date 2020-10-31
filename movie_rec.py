from tkinter import *
from tkinter import messagebox
import pandas as pd
from tkinter import ttk 
from ttkwidgets.autocomplete import AutocompleteCombobox
from surprise import SVD
from surprise import Dataset,Reader
from surprise.model_selection import cross_validate
from surprise import accuracy
from surprise.model_selection import train_test_split

movies=pd.read_csv('ml-100k/movies.csv')
ratings=pd.read_csv('ml-100k/ratings.csv')

def recommend():
    s.quit()
    movie=[i.get() for i in movlist]
    rat=[i.get() for i in tkvar]
    mid=[name_to_mid(i) for i in movie]
    # print(ids)
    # print(rats)
    rating=add_user_ratings(mid, rat,ratings)
    svd=calc_rating(rating)
    movs,r=rec_movies(movies, svd)
    s.destroy()
    win = Tk()
    win.title("Movies")
    win.geometry('400x300')
    win.configure(bg='tan2')
     
    print('test')
    Label(win,bg='tan2',fg='white',padx=5,pady=4, text="Recommended Movies:").grid(row=0, column=2)
    Label(win,bg='tan2',fg='white',padx=5,pady=4, text="Predicted rating:").grid(row=0, column=3)
    for i,j in enumerate(movs):
        Label(win,bg='tan2',fg='white',padx=5,pady=4, text=j).grid(row=i+1, column=2)
        Label(win,bg='tan2',fg='white',padx=5,pady=4, text=r[i]).grid(row=i+1, column=3)
    win.mainloop() 


def rec_movies(movies, svd):
    movies['est'] = movies['movieId'].apply(lambda x: svd.predict(672, x).est)
    movies = movies.sort_values('est', ascending=False)
    # movies.head(10)
    return (list(movies.head(10)['title']),list(movies.head(10)['est']))


def calc_rating(rating):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(
        ratings[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=.25)

    svd = SVD()

    svd.fit(trainset)
    return svd    

def add_user_ratings(mid, rat,ratings):
    new_user = pd.DataFrame({'userId': [672]*10,
                             'movieId': mid,
                             'rating': rat
                             })
    ratings=ratings.append(new_user)
    return ratings

def name_to_mid(movie):
    return int(movies.movieId[movies.title == movie])


s = Tk()
s.title("Movie Recommendation System")
s.geometry('600x800')
s.configure(bg='tan2')
movlist=[]
popupMenu=[]
tkvar=[]
Label(s, text="Rate previously seen movies for accurate predictions: ",fg='black',bg='tan2').grid(row=0, column=2)
for i in range(10):
    Label(s, text="Movie Name: ",fg='white',bg='tan2').grid(row=i+1, column=1)
    n=StringVar()
    movlist.append(AutocompleteCombobox(s, width = 27,textvariable =n)) 
    # movlist.append(ttk.Combobox(s, width = 27, textvariable = n) )
    movlist[i].set_completion_list(tuple(movies.title)) 
    # movlist[i]['values'] = tuple(movies.title)
    movlist[i].grid(column = 2, row = i+1) 
    choices = { 1,2,3,4,5}
    tkvar.append(StringVar(s))
    tkvar[i].set(3) # set the default option

    popupMenu.append(OptionMenu(s,  tkvar[i], *choices,))
    Label(s, text="Choose a rating: ",bg='tan2',fg='white').grid(row = i+1, column = 4)
    popupMenu[i].grid(row = i+1, column =5)
Button(s, text="Submit",bg='orange1',borderwidth=4,fg='white', command=recommend).grid(row=13, column=3,padx=5,pady=4)

s.mainloop()  
