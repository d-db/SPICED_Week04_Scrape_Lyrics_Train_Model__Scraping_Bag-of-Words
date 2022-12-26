#!/usr/bin/env python

# import all relevant scikit moduls
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import set_config

from bs4 import BeautifulSoup
from time import time

import requests
import re
import pandas as pd
import warnings

set_config(display='diagram')
warnings.filterwarnings('ignore')


def main():
    # Storing the current time in order to calculate the difference at the end of the program
    t0 = time()

    # Asking the user for the two artists and storing the URLs in a list
    artist_pages = list()
    for i in range(1, 3):
        artist_suffix = input(f"What is the page of your artist {i} - https://www.lyrics.com/artist/...?: ")
        artist_page = "https://www.lyrics.com/artist/" + artist_suffix
        artist_pages.append(artist_page)

    # Getting the artists name, all relative links that contain lyrics and the lyrics itself
    # Storing the return values in a list
    artist_lyrics = list()
    for i in range(0, 2):
        print("")
        print(f"The program is now scraping the lyrics of artist {i + 1}.\n"
              f"This may take a few minutes.")
        print("")
        lyrics_dic, artist_name, links = get_lyrics(artist_pages[i])
        artist_lyrics.append([len(links), artist_name, lyrics_dic])

    # Asking the user how many songs should be included in the corpus and storing the return values in a list
    print("")
    corpus_length = list()
    for i in range(0, 2):
        cl_artist = int(input(f"How many songs of {artist_lyrics[i][1]} do you want to include in the corpus? "
                              f"(max {artist_lyrics[i][0]}): "))
        corpus_length.append(cl_artist)

    # Constructing the corpus and the labels.
    lyrics_artist_1 = [value for value in artist_lyrics[0][2].values()][:corpus_length[0]]
    lyrics_artist_2 = [value for value in artist_lyrics[1][2].values()][:corpus_length[1]]
    corpus = lyrics_artist_1 + lyrics_artist_2
    label = [artist_lyrics[0][1]] * corpus_length[0] + [artist_lyrics[1][1]] * corpus_length[1]

    print("")
    print("The program is now constructing a model and will use GridSearch\n"
          "in order to find the best one for this task.\n"
          "Afterwards the results for the train and test data will be displayed.\n"
          "This may take a few minutes\n")

    # Build the model and use GridSearch()
    countvectorize_tfidf_clf_gridsearch(corpus, label)

    print("The program is over. Thank you for using it!")
    print(f"Done in {round((time() - t0)/60, 2)} minutes.")

    return artist_lyrics


def get_lyrics(artist_page):
    # Get the relative link of all songs from the artist page, find songs that are listed more than once
    links, multiple = get_unique_links(artist_page)

    # If lyrics available, add them to a dic
    new_links, artist_name, dic_lyrics = download_save_lyrics(links)

    # print_stat
    print_stats(artist_name, links, new_links, multiple)

    return dic_lyrics, artist_name, new_links


def get_unique_links(link):
    """
    Generates a list with relative links to songs on lyrics.com
    and a list with titles that have been listed multiple times
    :param link: Link to artist page as string
    :return: two lists: 1. unique links to lyrics, 2. Titles of songs listed more than once
    """
    r = requests.get(link)
    pattern_1 = '"(/lyric\/.*?)"'
    matches_1 = re.findall(pattern=pattern_1, string=r.text)

    title = []
    unique_rel_links = []
    multiple = []

    for match in matches_1:
        pattern_2 = "/lyric\/[\d]*\/.*?\/(.*?)$"
        match_2 = re.findall(pattern=pattern_2, string=match)[0]
        if match_2 not in title:
            unique_rel_links.append(match)
            title.append(match_2)
        else:
            multiple.append(match_2)

    return unique_rel_links, multiple


def download_save_lyrics(list_of_links):
    artist_page = ""
    artist = ""
    dic = {}
    new_list = list_of_links.copy()
    no_text = list()

    for i in range(len(list_of_links)):
        link = "http://www.lyrics.com/" + list_of_links[i]
        r = requests.get(link)
        lyrics_page = BeautifulSoup(r.text, "html.parser")

        # Only proceed when the html id can be found
        if len(lyrics_page.findAll(id="lyric-body-text")) > 0:
            # Add the name of the artist during the first iteration
            if len(artist) == 0:
                artist = lyrics_page.find("div", class_="artist-thumb")["data-artist"]

            # Find the lyrics and add them to dic
            song_title = lyrics_page.find(id="lyric-title-text").text
            lyrics = lyrics_page.find(id="lyric-body-text").text.replace("\n", " ").replace("\r", " ")
            dic[song_title] = lyrics
        else:
            no_text.append(list_of_links[i])

    # Remove the songs without lyrics from the link list
    for song in no_text:
        new_list.remove(song)

    return new_list, artist, dic


def print_stats(artist, old_list, new_list, multiple_entries):
    print(f"Your chosen artist is: {artist}\n"
          f"Number of unique song entries found on lyrics.com: {len(old_list)}\n"
          f"Number of songs that were listed multiple times: {len(multiple_entries)}\n"
          f"Number of songs with lyrics: {len(new_list)}")


def countvectorize_tfidf_clf_gridsearch(corpus, labels):
    # Assign X and y
    X = corpus
    y = labels

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Set the parameters
    parameters_lr = {
        "clf__penalty": ("l1", "l2", "elasticnet", "none"),
    }

    parameters_rf = {
        "clf__max_depth": (3, 10, 100),
        "clf__n_estimators": (3, 10)
    }

    parameters_svc = {
        "clf__kernel": ("linear", "poly", "rbf", "sigmoid", "precomputed")
    }

    parameters_nb = {
        "clf__alpha": (0.001, 1, 10)
    }

    # Define a list with relevant models
    models = [
        LogisticRegression(),
        RandomForestClassifier(),
        SVC(),
        MultinomialNB()
    ]

    list_of_parameters = [
        parameters_lr,
        parameters_rf,
        parameters_svc,
        parameters_nb
    ]

    train_scores = list()
    test_scores = list()

    for i, model in enumerate(models):
        parameters = list_of_parameters[i]
        pipeline = Pipeline(
            [
                ("vect", CountVectorizer(stop_words="english", max_df=0.05)),
                ("tfidf", TfidfTransformer()),
                ("clf", model)
            ]
        )

        # Execute GridSearchCV for every model
        grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
        # Fit the model with the best hyperparameters
        grid_search.fit(X_train, y_train)
        # Calculate the accuracy for train and test data
        train_result = grid_search.score(X_train, y_train)
        test_result = grid_search.score(X_test, y_test)

        train_scores.append(round(train_result, 2))
        test_scores.append(round(test_result, 2))

    df = pd.DataFrame({"Train-Data": train_scores,
                       "Test-Data": test_scores},
                      index=["LogisticRegression", "RandomForest", "SVC", "Naive Bayes"])

    print("")
    print(df)
    print("")


if __name__ == '__main__':
    artist_details = main()