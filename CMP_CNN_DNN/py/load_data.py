from pandas import read_csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.datasets import load_boston as sk_boston,load_breast_cancer as sk_breast_cancer
import os

np.seterr(divide='ignore',invalid='ignore')
current_path = os.path.dirname(__file__)
data_path = current_path + '/dataset/'
pd.options.mode.chained_assignment = None
def load_by_argv(argv_str):
    dic = argv_str
    if (argv_str == 'blood'):
        X, y = load_blood()
    elif (argv_str == 'housing'):
        X, y = load_housing()
    elif (argv_str == 'housing_classify'):
        X, y = load_housing_as_Classify()
    elif (argv_str == 'breast_cancer'):
        X, y = load_breast_cancer()
    elif (argv_str == 'rain'):
        X, y = load_weather()
    elif (argv_str == 'diabetes'):
        X, y = load_diabetes()
    elif (argv_str == 'heart'):
        X, y = load_heart()
    elif (argv_str == 'car'):
        X, y = load_car()
    elif (argv_str == 'musk'):
        X,y = load_musk()
    elif (argv_str == 'sonar'):
        X,y = load_sonar()
    elif (argv_str == 'glass'):
        X,y = load_glass()
    elif (argv_str == 'bupa'):
        X,y = load_bupa()
    elif (argv_str == 'mushroom'):
        X,y = load_mushroom()
    elif (argv_str == 'gender'):
        X,y = load_gender()
    elif (argv_str == 'mobile'):
        X,y = load_mobile()
    elif (argv_str == 'drug'):
        X,y = load_drug()
    elif (argv_str == 'bank'):
        X,y = load_bank()
    elif (argv_str == 'dota'):
        X,y = load_dota()
    elif (argv_str == 'water'):
        X,y = load_water()
    elif (argv_str == 'development'):
        X,y = load_development()
    elif (argv_str == 'ionosphere'):
        X, y = load_ionosphere()
    elif (argv_str == 'monk'):
        X, y = load_monk()
    elif (argv_str == 'pima'):
        X, y = load_pima()
    elif (argv_str == 'wdbc'):
        X, y = load_wdbc()
    elif (argv_str == 'customer'):
        X, y = load_customer()
    elif (argv_str == 'DryBean'):
        X, y = load_DryBean()
    elif (argv_str == 'Hydrologic'):
        X, y = load_Hydrologic()
    elif (argv_str == 'voice'):
        X, y = load_voice()
    elif (argv_str == 'nba'):
        X, y = load_nba()
    elif (argv_str == 'rice'):
        X, y = load_rice()
    elif (argv_str=='income'):
        X, y = load_income()
    elif (argv_str=='fraud'):
        X, y = load_fraud()
    elif (argv_str=='term'):
        X, y = load_term()
    elif (argv_str=='heartDisease'):
        X, y = load_heartDisease()
    elif (argv_str=='hospital'):
        X, y = load_hospital()
    else:
        print("no such dataset")
        exit()
    X = linear_mapping(X)
    X = modify_feature_names(X)

    if isinstance (y,np.ndarray):
        y = array_to_Series(y)
    return X,y,dic


######################2021-08-20 ################
def load_ionosphere():
    # uci dataset  https://archive-beta.ics.uci.edu/ml/datasets/52
    global data_path
    data = read_csv(data_path + 'ionosphere.data',header=None)
    encode_df(data)
    X = data.drop([1,34],axis=1)
    y = data[34]
    return X,y
def load_wdbc():
    global data_path
    data = read_csv(data_path + 'wdbc.data',header=None)
    encode_df(data)
    X = data.drop(1,axis=1)
    y = data[1]
    return X,y
def load_pima():
    # kaggle https://www.kaggle.com/uciml/pima-indians-diabetes-database
    global data_path
    data = read_csv(data_path + 'pima.csv')
    X = data.drop('Outcome',axis=1)
    y = data['Outcome']
    return X,y
def load_monk():
    # uci https://archive-beta.ics.uci.edu/ml/datasets/70
    global data_path
    data = read_csv(data_path + 'monk.data', header=None,sep=' ')
    X = data.drop([0,1,8], axis=1)
    y = data[1]
    return X,y
####################### end #####################

def load_blood():
    global data_path
    data = read_csv(data_path + 'transfusion.data')
    X = data.iloc[: , :-1]
    y = data['whether he/she donated blood in March 2007']
    return X,y  


def load_musk():
    global data_path
    data = read_csv(data_path + 'clean2.data',header=None)
    X = data.drop([0,1,168],axis=1)
    y=data[0]
    y_tmp = y.copy()
    for i ,element in enumerate(y):
        y_tmp[i]=element.split('-')[0]
    y =y_tmp
    le = LabelEncoder()
    y=le.fit_transform(y)
    return X,y

def load_weather():
    global data_path
    data = read_csv(data_path + "weatherAUS.csv")
    cols_to_drop = ["Date", "Location", "RainTomorrow", "Rainfall"]
    data.drop(cols_to_drop, axis=1, inplace=True)
    x = data.isna()
    missing_props = data.isna().mean(axis=0)
    over_threshold = missing_props[missing_props >= 0.4]
    data.drop(over_threshold.index,axis=1,inplace=True)
    X = data.drop("RainToday", axis=1)
    y = data.RainToday
    X = data.drop("RainToday", axis=1)
    y = data.RainToday
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("oh-encode", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    from sklearn.preprocessing import StandardScaler

    numeric_pipeline = Pipeline(
        steps=[("impute", SimpleImputer(strategy="mean")),
               ("scale", StandardScaler())]
    )
    cat_cols = X.select_dtypes(exclude="number").columns
    num_cols = X.select_dtypes(include="number").columns

    from sklearn.compose import ColumnTransformer

    full_processor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, num_cols),
            ("categorical", categorical_pipeline, cat_cols),
        ]
    )
    X_processed = full_processor.fit_transform(X)
    y_processed = SimpleImputer(strategy="most_frequent").fit_transform(
        y.values.reshape(-1, 1)
    )
    X_processed = pd.DataFrame(X_processed)
    print(type(X_processed))
    le = LabelEncoder()
    y=le.fit_transform(y_processed)
    print(type(y))
    return X_processed,y

def load_sonar():
    global data_path
    data = read_csv(data_path + 'sonar.all-data',header=None)
    X=data.drop(60,axis=1)
    y=data[60]
    le = LabelEncoder()
    y=le.fit_transform(y)
    return X,y
def load_gender():
    global data_path
    data = read_csv(data_path + 'gender.csv')
    data = encode_df(data)
    X = data.drop('Gender',axis=1)
    y = data['Gender']
    return X,y
#mobile :multi classes to 2 classes
def load_mobile():
    global data_path
    data = read_csv(data_path + 'mobile.csv')
    data = encode_df(data)
    most_class = data[data['price_range'] == 0]
    second_most_class = data[data['price_range'] == 1]
    top_two = pd.concat([most_class, second_most_class])
    X = top_two.drop('price_range', axis=1)
    y = top_two['price_range']
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    return X, y
#drug : multi classes to 2 classes
def load_drug():
    global data_path
    data = read_csv(data_path + 'drug200.csv')
    data = encode_df(data)
    most_class = data[data['Drug'] == 0]
    second_most_class = data[data['Drug'] == 4]
    top_two = pd.concat([most_class, second_most_class])
    X = top_two.drop('Drug', axis=1)
    y = top_two['Drug']
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    y[y == 4] = 1
    return X,y
def load_mushroom():
    #8124 * 23
    global data_path
    data = read_csv(data_path + 'mushrooms.csv')
    data = encode_df(data)
    y = data['class']
    X = data.drop('class', axis=1)
    # print(data.info())
    return X,y
# car :3 classes to 2 classes
def load_car():
    global data_path
    data = read_csv(data_path + 'car.data', header=None)
    data = encode_df(data)

    most_class = data[data[6] == 0]
    second_most_class = data[data[6] == 2]
    top_two = pd.concat([most_class, second_most_class])
    X = top_two.drop(6, axis=1)
    y = top_two[6]
    y[y == 2] = 1
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    return X, y
# glass : multiclass to 2 classes
def load_glass():
    global data_path
    data = read_csv(data_path + 'glass.data',header=None)
    most_class = data[data[10] == 1]
    second_most_class = data[data[10] == 2]
    top_two = pd.concat([most_class,second_most_class])
    X = top_two.drop(10, axis=1)
    y = top_two[10]
    y[y == 1] = 0
    y[y == 2] = 1
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    return X, y

def array_to_Series(y):
    y_series = pd.Series(y.squeeze())
    return y_series
        
def modify_feature_names(X):
    new_names ={}
    for i,element in enumerate(X.columns):
        new_name = 'f'+ str(i)
        new_names[element] = new_name
    T = X.rename(columns = new_names)
#     print(T.columns)
    return T

def linear_mapping(X):
    X_arr = np.array(X)
    minVals = X_arr.min(0)
    maxVals = X_arr.max(0)
    ranges = maxVals - minVals
    X_norm = np.zeros(X_arr.shape)
    m=X_arr.shape[0]
    X_norm = X_arr -np.tile(minVals,(m,1))
    X_norm = 15.9 * X_norm/np.tile(ranges,(m,1))
    X = pd.DataFrame(X_norm)
    return X

def encode_df(dataframe):
    le = LabelEncoder()
    for column in dataframe.columns:
        dataframe[column] = le.fit_transform(dataframe[column])
    return dataframe

########## 8-13 ################

def load_dota():
    global data_path
    data = read_csv(data_path + 'dota_games.csv',header=None)
    data = encode_df(data)
    y = data[data.shape[1]-1]
    X = data.drop(data.shape[1]-1,axis=1)
    return X,y
def load_bank():
    global data_path
    data = read_csv(data_path + 'bank.csv')
    data = encode_df(data)
    y = data.y
    X = data.drop('y', axis=1)
    return X, y
def load_water():
    global data_path
    data = read_csv(data_path + 'waterQuality1.csv')
    # data.info()
    X = data.drop('is_safe',axis=1)
    y = data.is_safe
    return X,y
def load_development():
    global data_path
    data = read_csv(data_path + 'Development_Index.csv')
    most_class = data[data['Development Index'] == 3]
    second_most_class = data[data['Development Index'] == 4]
    top_two = pd.concat([most_class, second_most_class])
    top_two=top_two.reset_index(drop=True)
    # return top_two
    X = top_two.drop('Development Index',axis=1)
    y = top_two['Development Index']
    y[y == 3] = 0
    y[y == 4] = 1
    return X,y

##############################################################

def load_heart():
    global data_path
    heart = loadmat(data_path + "Heartstatlog.mat")

    y = heart['Heartstatlog'][:, 0]
    X_org = heart['Heartstatlog'][:, 1:]

    X = X_org[:, 0:2]
    X = np.concatenate((X, X_org[:, 3:6]), axis=1)
    X = np.concatenate((X, X_org[:, 7:12]), axis=1)
    for i in [2, 6, 12]:
        label_encoder = LabelEncoder()
        feature = label_encoder.fit_transform(X_org[:, i])
        feature = feature.reshape(X_org.shape[0], 1)
        onehot_encoder = OneHotEncoder(sparse=False)
        feature = onehot_encoder.fit_transform(feature)
        if X is None:
            X = feature
        else:
            X = np.concatenate((X, feature), axis=1)

    for t in enumerate(y):
        if t[1] == 1:
            y[t[0]] = 0
        else:
            y[t[0]] = 1

    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    return X,y

def load_bupa():
    global data_path
    bupa = loadmat(data_path+"bupa.mat")

    y = bupa['Bupa'][:, 0]
    X = bupa['Bupa'][:, 1:]

    for t in enumerate(y):
        if t[1] == 1:
            y[t[0]] = 0
        else:
            y[t[0]] = 1
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    return X,y

def load_diabetes():
    global data_path
    diabetes = loadmat(data_path + "diabetes.mat")

    y = diabetes['Diabetes'][:, 0]
    X = diabetes['Diabetes'][:, 1:]

    for t in enumerate(y):
        if t[1] == 1:
            y[t[0]] = 0
        else:
            y[t[0]] = 1

    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    return X,y

def load_housing():
    housing = sk_boston()
    X = housing.data
    y = housing.target
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    return X,y

def load_housing_as_Classify():
    housing = sk_boston()
    X = housing.data
    y = housing.target
    for i in range(len(y)):
        if y[i] >= 25:
            y[i]=1
        else :
            y[i]=0
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    return X,y

def load_breast_cancer():
    cancer = sk_breast_cancer()
    X = cancer.data
    y = cancer.target
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    return X,y


##################################################################
# added at 2021 12 06
##################################################################

def load_customer():
    global data_path
    data = read_csv(data_path + 'customer/customer_data.csv')
    X = data.drop('label',axis=1)
    y = data.label
    return X, y

def load_DryBean():
    global data_path
    data = read_csv(data_path + 'DryBeanDataset/Dry_Bean_Dataset.csv')
    data = encode_df(data)

    most_class = data[data['Class']==3]
    second_class = data[data['Class']==6]
    top_two = pd.concat([most_class, second_class])

    X = top_two.drop('Class', axis=1)
    y = top_two.Class
    y[y == 3] = 0
    y[y == 6] = 1
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    return X, y

def load_Hydrologic():
    global data_path
    data = read_csv(data_path + 'Summarized_HUC08_HydrologicClass_v1.csv')  
    data = encode_df(data)

    most_class = data[data['NUMCLASS']==1]
    second_class = data[data['NUMCLASS']==2]
    top_two = pd.concat([most_class, second_class])

    X = top_two.drop('NUMCLASS', axis=1)
    y = top_two.NUMCLASS
    y[y == 1] = 0
    y[y == 2] = 1
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    return X, y

def load_voice():
    global data_path
    data = read_csv(data_path + 'ml-research-gender-recognition-by-voice/voice.csv')
    data = encode_df(data)
    X = data.drop('label',axis=1)
    y = data.label
    return X, y

def load_nba():
    global data_path
    data = read_csv(data_path + 'exercises-logistic-regression-exercise-1/nba_logreg.csv')
    X = data.drop('TARGET_5Yrs',axis=1).drop('Name',axis=1)
    y = data.TARGET_5Yrs
    return X, y

def load_rice():
    global data_path
    data = read_csv(data_path + 'riceClassification.csv')
    X = data.drop('Class',axis=1).drop('id',axis=1)
    y = data.Class
    return X, y
##################################################################
# added at 2022 03 15
##################################################################
def load_income():
    global data_path
    data = pd.read_csv(data_path+'income.csv')
    data = encode_df(data)
    X = data.iloc[:, :-1]
    y = data.iloc[:,-1]
    return X,y
def load_fraud():
    global data_path
    data = pd.read_csv(data_path+'fraud.csv')
    X = data.iloc[:,1:-1]
    y = data.iloc[:,-1]
    return X, y
def load_term():
    global data_path
    data = pd.read_csv(data_path+'term.csv')
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y
def load_heartDisease():
    global data_path
    data = pd.read_csv(data_path+'heartDisease.csv')
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    return X, y

######## 20220321 #############
def load_hospital():
    global data_path
    data = pd.read_csv(data_path+'hospital.csv')
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y