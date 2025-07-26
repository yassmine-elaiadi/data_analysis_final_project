import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'

df = pd.read_csv(path)
df.head()
 
##########################################################QUESTION 1 
print(df.dtypes)
print(df.describe())
##########################################################QUESTION 2
df.drop(['id', 'Unnamed: 0'] , axis= 1 , inplace= True)
print(df.describe())
print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())

mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)
mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace=True)
print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())

############################################################QUESTION 3

flour_count = df['floors'].value_counts()
flour_count_df = flour_count.to_frame()
print(flour_count_df)

#############################################################QUESTION 4

sns.boxplot(x="waterfront" , y="price", data=df)
plt.show()

#############################################################QUESION 5
sns.regplot(x= "sqft_above" , y="price" ,data= df )
df_numeric = df.select_dtypes(include=[np.number])
df_numeric.corr()['price'].sort_values()
plt.show()

#############################################################QUESTION 6 
X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
r= lm.score(X, Y)
print("R^2 = ", r)
X = df[['sqft_living']]
Y =df['price']
lm.fit(X,Y)
r2=lm.score(X,Y)
print("R^2 = ", r2)

##############################################################QUESTION 7
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]     
X=df[features]
Y= df['price']
lm = LinearRegression()
lm.fit(X,Y)
r3 = lm.score(X,Y)
print("R^2 = ", r3)

##############################################################QUESTION 8

# Create the list of tuples for the pipeline
Input = [
    ('scale', StandardScaler()),
    ('polynomial', PolynomialFeatures(include_bias=False)),
    ('model', LinearRegression())
]

pipe = Pipeline(Input)
X=df[features]
Y= df["price"]
pipe.fit(X,Y)
r4 = pipe.score(X,Y)
print("R^2 = " , r4)

###############################################################QUESTION 9
X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=1)


print("number of test samples:", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

ridge_model = Ridge(alpha=0.1)
ridge_model.fit(x_train, y_train)

r5 = ridge_model.score(x_test, y_test)
print("R^2 = " , r5)

########################################################QUESTION 10

poly = PolynomialFeatures(degree= 2 , include_bias=False)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.fit_transform(x_test)

ridge_model = Ridge(alpha=0.1)
ridge_model.fit(x_train_poly, y_train)
r_poly = ridge_model.score(x_test_poly,y_test)
print("r_poly = " , r_poly)

