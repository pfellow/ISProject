import pickle
import socket
import warnings

import numpy as np
import pandas as pd
import rdflib
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


# Defining functions
def target_encoding(df, column, base, fillna='None'):
    df[base] = df[base].replace({'None': np.NaN})
    if df[column].isnull().sum() >= 1:
        print('Number of NaN values: ' + str(df[column].isnull().sum()))
        print('<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>')
        df[column] = df[column].fillna(fillna)
    encodings = df.groupby(column)[base].mean().reset_index().rename(columns={base: column + 'Encoded'})
    print(encodings)
    print('<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>')
    df = df.merge(encodings, how='left', on=column)
    df.drop(column, axis=1, inplace=True)
    print(df.info())
    return df


def group_categories(df, column, number):
    frequency = df[column].value_counts().reset_index()
    print(frequency)
    print('<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>')

    for i in range(len(df[column])):
        ind = frequency.loc[frequency['index'] == df[column][i]].index[0]
        if frequency.iloc[ind][1] < number:
            df[column][i] = 'Other'
    print(df[column].value_counts().reset_index())
    return df


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('localhost', 1234))
s.listen(1)

while True:
    conn, addr = s.accept()
    print("Connection to a client has been established")
    conn.sendall(bytes("Connection to the server has been established. \n Processing the input data...", 'utf-8'))
    filename = conn.recv(1024).decode('utf-8')
    print(filename)
    startup_data = pd.read_csv(filename)
    print(startup_data)

    # PREPROCESSING

    print(startup_data.isnull().sum())

    # Replacing text statuses

    startup_data["status"] = startup_data["status"].replace({"acquired": 1, "closed": 0})

    # state_code

    print(startup_data.groupby("state_code")['status'].mean().reset_index())
    group_categories(startup_data, 'state_code', 2)
    startup_data = target_encoding(startup_data, 'state_code', 'status')

    # zip_code

    print(list(set(startup_data.zip_code)))

    for i in range(len(startup_data['zip_code'])):
        if ' ' in startup_data['zip_code'][i]:
            startup_data['zip_code'][i] = startup_data['zip_code'][i].split(' ')[1]
        if '-' in startup_data['zip_code'][i]:
            startup_data['zip_code'][i] = startup_data['zip_code'][i].split('-')[0]

    group_categories(startup_data, 'zip_code', 2)
    startup_data = target_encoding(startup_data, 'zip_code', 'status')

    # city

    group_categories(startup_data, 'city', 2)
    startup_data = target_encoding(startup_data, 'city', 'status')

    # category_code

    print(startup_data["category_code"])
    group_categories(startup_data, 'category_code', 2)
    startup_data = target_encoding(startup_data, 'category_code', 'status')

    # dropping unnecessary columns

    # column "labels" is equal to the target "status", so it needs to be dropped

    g = rdflib.Graph()
    result = g.parse(file=open("StartupSuccessKB.n3", "r+"), format="text/n3")

    columns_to_drop = [];

    qres = g.query(
        """SELECT DISTINCT ?col
            WHERE {
                ind:proc12 prop:columnToDelete ?col.
            }""")

    for row in qres:
        columns_to_drop.append(str(row.asdict()['col'].toPython()))

    print('<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>')
    print("Columns to be excluded:")

    for col in columns_to_drop:
        print(col)
    print('<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>')

    startup_data.drop(columns_to_drop, axis=1, inplace=True)
    print(startup_data.isnull().sum())

    startup_data["age_first_milestone_year"] = startup_data["age_first_milestone_year"].fillna(0)
    startup_data["age_last_milestone_year"] = startup_data["age_last_milestone_year"].fillna(0)

    startup_data.describe()

    # MinMaxScaler

    for i in ["age_first_funding_year", "age_last_funding_year", "relationships", "funding_rounds", "funding_total_usd",
              "milestones", "avg_participants"]:
        values = startup_data[i].values
        values = values.reshape(len(values), 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(values)
        normalized = scaler.transform(values)
        startup_data[i] = normalized

    # Splitting train and test data

    X = startup_data.drop('status', axis=1)
    y = startup_data['status']

    qres1 = g.query(
        """SELECT DISTINCT ?testVolume
            WHERE {
                ind:proc14 prop:testVolume ?testVolume.
            }""")

    for value in qres1:
        test_size = value.asdict()['testVolume'].toPython()

    conn.sendall(bytes("Current size of the test data is " + str(test_size) + "%.", 'utf-8'))

    test_size_new = conn.recv(1024).decode('utf-8')

    if test_size_new == "0":
        conn.sendall(bytes("Using the default size of the test data " + str(test_size) + "%.", 'utf-8'))
        test_size = test_size / 100
    elif 5 <= int(test_size_new) <= 50:
        result.update(
            """
            DELETE {
                ind:proc14 prop:testVolume ?testVolume 
            }
            INSERT {
                ind:proc14 prop:testVolume xxx
            }
            WHERE {
                ind:proc14 prop:testVolume ?testVolume
            }""".replace("xxx", test_size_new)
        )

        test_size = int(test_size_new) / 100
        conn.sendall(bytes("The size of the test data has been changed to " + str(test_size_new) + "%.", 'utf-8'))
    else:
        conn.sendall(bytes("Incorrect", 'utf-8'))

    conn.sendall(bytes("Training and evaluating the neuronet...", 'utf-8'))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)

    # # NEURONET TRAINING AND PREDICTION

    model = Sequential()
    model.add(Dense(24, activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_results = model.fit(X_train, y_train, epochs=200, batch_size=64, validation_data=(X_test, y_test), verbose=0)

    aver_acc = np.mean(model_results.history["val_accuracy"])
    print("Keras Neuronet Accuracy: ", aver_acc)

    y_pred = model.predict(X_test) > 0.5

    # sending data for plots

    acc = model_results.history['accuracy']
    val = model_results.history['val_accuracy']

    conn.sendall(pickle.dumps(acc))
    conn.sendall(pickle.dumps(val))

    conn.sendall(pickle.dumps(y_test))
    conn.sendall(pickle.dumps(y_pred))

    # Writing new data to KB

    result.update(
        """
        DELETE {
            ind:proc04 prop:neuronet_accuracy ?old_result
        }
        INSERT {
            ind:proc04 prop:neuronet_accuracy new_result
        }
        WHERE {
            ind:proc04 prop:neuronet_accuracy ?old_result
        }""".replace("new_result", str(aver_acc))
    )
    result.serialize(destination="StartupSuccessKB.n3")

conn.close()
