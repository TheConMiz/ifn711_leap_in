from io import StringIO
import numpy as np
import pandas as pd
import pydot
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz


def data_prep():

    df = pd.read_csv('D3.csv')

    ######################         Data Preprocessing      #########################

    #imputing missing values with the median 
    df['contacts_count'] = df['contacts_count'].fillna(df['contacts_count'].median())
    df['house_count'] = df['house_count'].fillna(df['house_count'].median())
    df['public_transport_count'] = df['public_transport_count'].fillna(df['public_transport_count'].median())
    df['worried'] = df['worried'].fillna(df['worried'].median())

    # imputing missing values with the most frequent value (mode)
    workingmode = df['working'].value_counts().index[0]
    df['working'].fillna(workingmode, inplace = True)
    insurancemode = df['insurance'].value_counts().index[0]
    df['insurance'].fillna(insurancemode, inplace = True)
    incomemode = df['income'].value_counts().index[0]
    df['income'].fillna(incomemode, inplace = True)
    racemode = df['race'].value_counts().index[0]
    df['race'].fillna(racemode, inplace = True)
    immigrantmode = df['immigrant'].value_counts().index[0]
    df['immigrant'].fillna(immigrantmode, inplace = True)
    smokingmode = df['smoking'].value_counts().index[0]
    df['smoking'].fillna(smokingmode, inplace = True)

    # coverting data types of the variables 
    df['contacts_count'] = df['contacts_count'].astype(int)
    df['worried'] = df['worried'].astype(bool)
    df['covid19_positive'] = df['covid19_positive'].astype(bool)
    df['covid19_symptoms'] = df['covid19_symptoms'].astype(bool)
    df['covid19_contact'] = df['covid19_contact'].astype(bool)
    df['asthma'] = df['asthma'].astype(bool)
    df['kidney_disease'] = df['kidney_disease'].astype(bool)
    df['liver_disease'] = df['liver_disease'].astype(bool)
    df['compromised_immune'] = df['compromised_immune'].astype(bool)
    df['heart_disease'] = df['heart_disease'].astype(bool)
    df['lung_disease'] = df['lung_disease'].astype(bool)
    df['diabetes'] = df['diabetes'].astype(bool)
    df['hiv_positive'] = df['hiv_positive'].astype(bool)
    df['hypertension'] = df['hypertension'].astype(bool)
    df['other_chronic'] = df['other_chronic'].astype(bool)
    df['nursing_home'] = df['nursing_home'].astype(bool)
    df['health_worker'] = df['health_worker'].astype(bool)

    #imputing outliers in height by median of the height data
    df['height'] = np.where(df['height'] > 205, df['height'].median(), df['height'])
    df['height'] = np.where(df['height'] < 140, df['height'].median(), df['height'])

    #imputing outliers in weight by median of the weight data 
    df['weight'] = np.where(df['weight'] > 129, df['weight'].median(), df['weight'])

    # one-hot encoding
    df = pd.get_dummies(df)

    # target/input split
    y = df['covid19_positive']
    X = df.drop(['covid19_positive'], axis=1)

    # setting random state
    rs = 10
    X_mat = X.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.3, stratify=y, random_state=rs)

    return df,X,y,X_train, X_test, y_train, y_test

def analyse_feature_importance(dm_model,feature_names,n_to_display=20):
    # grab feature importances from the model
    importances = dm_model.feature_importances_
    
    # sort them out in descending order
    indices = np.argsort(importances)
    indices = np.flip(indices, axis=0)

    # limit to 20 features, you can leave this out to print out everything
    indices = indices[:n_to_display]

    for i in indices:
        print(feature_names[i], ':', importances[i])

def visualize_decision_tree(dm_model, feature_names, save_name):
    dotfile = StringIO()
    export_graphviz(dm_model, out_file=dotfile, feature_names=feature_names)
    graph = pydot.graph_from_dot_data(dotfile.getvalue())
    graph[0].write_png(save_name) # saved in the following file
