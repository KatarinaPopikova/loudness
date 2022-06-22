import numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as seaborn
from sklearn import preprocessing
from collections import Counter

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVR
from sklearn.datasets import make_regression
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import AdaBoostClassifier


def print_describe(data):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(data.describe())


def print_plot(data_train, col):
    plt.scatter(data_train[col], data_train['loudness'])
    plt.xlabel(col)
    plt.ylabel('loudness')
    plt.show()


def show_matrix(data_train):
    plt.figure(figsize=(15, 12))
    plt.xticks(rotation=90)
    seaborn.heatmap(data_train.corr())
    plt.show()


def show_outliers_plot(data):
    plt.figure(figsize=(15, 12))
    plt.xticks(rotation=90)
    data.boxplot()
    plt.show()


def show_all_plots(data_train):
    print_plot(data_train, 'explicit')
    print_plot(data_train, 'danceability')
    print_plot(data_train, 'energy')
    print_plot(data_train, 'key')
    print_plot(data_train, 'mode')
    print_plot(data_train, 'speechiness')
    print_plot(data_train, 'acousticness')
    print_plot(data_train, 'instrumentalness')
    print_plot(data_train, 'liveness')
    print_plot(data_train, 'valence')
    print_plot(data_train, 'tempo')
    print_plot(data_train, 'artist_followers')


def show_describe(data_train):
    show_all_plots(data_train)
    show_matrix(data_train)
    print_describe(data_train)


def print_duplicate(data):
    print("\n\nPočet duplicitných riadkov:")
    print(data.duplicated(subset=['artist', 'name']).sum())
    print("\n\n")


# Press the green button in the gutter to run the script.
def change_date_to_year_number(data):
    data['year'] = pd.DatetimeIndex(data['release_date']).year
    return data


def remove_outlier(data, col_name):
    q1 = data[col_name].quantile(0.25)
    q3 = data[col_name].quantile(0.75)
    iqr = q3 - q1  # Interquartile range
    fence_low = q1 - 1.5 * iqr
    fence_high = q3 + 1.5 * iqr
    df_out = data.loc[(data[col_name] > fence_low) & (data[col_name] < fence_high)]
    return df_out


def handle_outliers(data):
    # show_outliers_plot(data)
    data = remove_outlier(data, 'tempo')
    data = remove_outlier(data, 'year')
    return data


def edit_data(data, train):
    # drop unnecessary cols
    data = data.drop(['artist_followers', 'duration_ms', 'popularity', 'explicit', 'key', 'mode', 'speechiness'],
                     axis='columns')
    # print_describe(data)

    # remove duplicate rows in train
    if train:
        # print_duplicate(data)
        data = data.drop_duplicates(subset=['artist', 'name'])
        # print_describe(data)

    data = change_date_to_year_number(data)
    # print_describe(data)

    if train:
        data = handle_outliers(data)
        # print_describe(data)

    data = data.reset_index(drop=True)
    return data


def cut(data):
    # split dataset into categorical, numerical, result
    cat_cols = list(data.dtypes[data.dtypes == 'object'].index.values)
    cat_cols.remove('artist_genres')
    data.drop(cat_cols, axis='columns', inplace=True)
    num_cols = list(data.dtypes[data.dtypes != 'object'].index.values)
    num_data = data.copy()
    num_data.drop(['loudness', 'artist_genres'], axis='columns', inplace=True)
    cat_data = data.copy()
    cat_data.drop(num_cols, axis='columns', inplace=True)
    num_cols.remove('loudness')
    cat_cols.append('artist_genres')
    y = data['loudness']
    # y.drop(num_cols + ['artist_genres'], axis='columns', inplace=True)
    return [y, num_data, cat_data]


def manage_categorical(genres_col, famous_artist_genres):
    # change categorical into dataset
    for genre in famous_artist_genres:
        col_values = []
        for i in range(genres_col.shape[0]):
            if genres_col['artist_genres'][i].find(genre) > 0:
                col_values.append(1)
            else:
                col_values.append(0)
        genres_col[genre] = col_values
    genres_col.drop('artist_genres', axis='columns', inplace=True)

    # print(genres_col.head())
    return genres_col


def find_famous_artist_genres(data):
    all_artist_genres = []
    famous_genres = []
    count = 15

    # change string into array
    for i in range(data.shape[0]):
        all_artist_genres.extend(
            data['artist_genres'][i].replace('\'', '').replace(']', '').replace('[', '').split(", "))
    c = Counter(all_artist_genres)
    most_commom = c.most_common(count)
    # print(most_commom)
    for i in range(count):
        famous_genres.append(most_commom[i][0])
    return famous_genres


def manage_null_values(data, data_train):
    null_columns = data.columns[data.isnull().any()]
    # print(data[null_columns].isnull().sum())
    # null replace with mean
    for i in null_columns:
        data[i] = data[i].fillna(data_train[i].mean())
    return data


def min_max_normalization(data_test, data_train):
    # Min-Max Normalization

    scaler = preprocessing.MinMaxScaler()
    names = data_test.columns
    d = scaler.fit_transform(data_train)

    scaled_data_train = pd.DataFrame(d, columns=names)

    d = scaler.transform(data_test)
    scaled_data_test = pd.DataFrame(d, columns=names)

    return scaled_data_train, scaled_data_test


def print_plot_reziduals(reziduals, y, col):
    plt.scatter(y, reziduals)
    plt.xlabel('loudness')
    plt.ylabel(col)
    plt.show()


def manage_reziduals(predict_data_test, y_test):
    # print('loudness:')
    # print(y_test.head())
    # print('predict loudness:')
    # print(predict_data_test[:5])
    reziduals_test = y_test - predict_data_test
    print(reziduals_test.head())
    print_describe(reziduals_test)
    print_plot_reziduals(reziduals_test, list(range(1, len(reziduals_test) + 1)), 'rezidualy_test')


def regression(regr, train_data_new, y_train, test_data_new, y_test, what):
    regr.fit(train_data_new, y_train.values)
    predict_data_test = regr.predict(test_data_new)
    predict_data_train = regr.predict(train_data_new)

    print(what + ":\n")
    mse_train = (mean_squared_error(y_train, predict_data_train))
    mse_test = (mean_squared_error(y_test, predict_data_test))
    print("Train MSE: %f   ,   Test MSE:  %f" % (mse_train, mse_test))

    print("\n")

    r2_train = (r2_score(y_train, predict_data_train))
    r2_test = (r2_score(y_test, predict_data_test))
    print("Train R2: %f   ,   Test R2:  %f" % (r2_train, r2_test))

    if what == 'GridSearch':
        print("Najlepšie parametre pre SVR: ")
        print(regr.best_params_)


    # manage_reziduals(predict_data_test, y_test)


def cross_valid(train_data_new, test_data_new, y_train, y_test):
    scores = []
    regressor = SVR(C=1, kernel="rbf", gamma="scale")
    k = KFold(n_splits=6, random_state=42, shuffle=True)
    for i_train, i_test in k.split(train_data_new):
        X_train1, X_valid1, y_train1, y_valid1 = train_data_new.iloc[i_train], train_data_new.iloc[i_test], \
                                                 y_train.iloc[i_train], y_train.iloc[i_test]
        regressor.fit(X_train1, y_train1.values.ravel())
        predict_valid = regressor.predict(X_valid1)
        scores.append(r2_score(y_valid1, predict_valid))

    predict_test = regressor.predict(test_data_new)
    predict_train = regressor.predict(train_data_new)

    print("\n\nVal R2 (mean): ", str(numpy.mean(scores)))
    print("Train MSE: %f   ,   Test MSE:  %f" % (
        mean_squared_error(y_train, predict_train), mean_squared_error(y_test, predict_test)))
    print("Train R2: %f   ,   Test R2:  %f" % (r2_score(y_train, predict_train), r2_score(y_test, predict_test)))


def regresors(train_data_new, test_data_new, y_train, y_test):
    regresor1 = SVR(C=10.0, gamma="scale", kernel="rbf")
    regression(regresor1, train_data_new, y_train, test_data_new, y_test, 'SVR')

    # regresor2 = BaggingRegressor(base_estimator=SVR(), n_estimators=10, random_state=0)
    # regression(regresor2, train_data_new, y_train, test_data_new, y_test, 'Bagging')

    # cross_valid(train_data_new, test_data_new, y_train, y_test)

    # regresor3 = AdaBoostRegressor(n_estimators=100, random_state=0)
    # regression(regresor3, train_data_new, y_train, test_data_new, y_test, 'AdaBoost')

    # parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    # svr = SVR()
    # regresor4 = GridSearchCV(svr, parameters)
    # regression(regresor4, train_data_new, y_train, test_data_new, y_test, 'GridSearch')


if __name__ == '__main__':
    pd.set_option('display.width', 360)

    data_test = pd.read_csv("data/spotify_test.csv")
    data_train = pd.read_csv("data/spotify_train.csv")

    show_describe(data_train)

    # drop unnecessary cols     # remove duplicate rows
    data_train = edit_data(data_train, True)
    data_test = edit_data(data_test, False)

    # split categorical/numerical data and result
    [y_train, num_train, cat_train] = cut(data_train)
    [y_test, num_test, cat_test] = cut(data_test)

    # replace null values with mean
    num_train = manage_null_values(num_train, num_train)
    num_test = manage_null_values(num_test, num_train)

    # normalize only numerical data from dataset
    [scaled_data_test, scaled_data_train] = min_max_normalization(num_train, num_test)
    #
    print_describe(scaled_data_train)
    print_describe(scaled_data_test)

    # find most popular genres (15) and make categorical dataset
    famous_artist_genres = find_famous_artist_genres(cat_train)
    cat_train = manage_categorical(cat_train, famous_artist_genres)
    cat_test = manage_categorical(cat_test, famous_artist_genres)
    # print_describe(cat_train)
    # print_describe(cat_test)

    # connect numerical dataset with categorical
    train_data_new = pd.concat([scaled_data_train, cat_train], axis=1)
    test_data_new = pd.concat([scaled_data_test, cat_test], axis=1)

    print_describe(train_data_new)
    print_describe(test_data_new)

    regresors(train_data_new, test_data_new, y_train, y_test)
