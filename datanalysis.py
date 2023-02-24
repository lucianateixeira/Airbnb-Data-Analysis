#!/usr/bin/env python
# coding: utf-8

# **CCT - Final Project**

# In[1]:


# Importing the libraries
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, LabelEncoder
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesRegressor
from __future__  import print_function #adds compatibility with new versions of Python
get_ipython().run_line_magic('matplotlib', 'inline')
#it needs to have two underscore for 'future', if not it thorws an error: 
#ModuleNotFoundError: No module named '_future_'

import warnings
warnings.filterwarnings("ignore") #This command ignores the warning 


# In[2]:


#Dependecies used in this notebook.
#To install, please remove the hashtag.

#pip install panda
#pip install seaborn
#pip install plotly
#pip install matplotlib
#pip install scikit-learn
#pip install spacy
#pip install wordcloud
#pip install stopwords
#pip install tensorflow
#pip install googletrans==4.0.0rc1


# Index of Datasets:
# <p> 1) Listings </p>
# <p> 2) Review </p>
# <p> 3) Calendar </p>
# <p> 4) Airbnb_listing </p>
# <p> 5) Airbnb_listing1 </p>
# <p> 6) Rate </p>
# 
# <p>Some of the datasets are for trends and whether or not features influence reviews that influence occupancy that generates higher ROI. 
# the dataset for ML is the last one, it is a sample data of Dublin, I want to identify the occupancy rate in the future.</p>

# In[3]:


#Importing the dataset #1

#The dataset chosen for this project is listings for Airbnb 

listing = pd.read_csv('/Users/lucianateixeira/Desktop/listings.csv')


# In[4]:


#Importing the dataset #2

#The dataset chosen for this project is reviews for Airbnb 

review = pd.read_csv('/Users/lucianateixeira/Desktop/reviews.csv')


# In[5]:


#Importing the dataset # 3

#The dataset chosen for this project is calendar for Airbnb

calendar = pd.read_csv('/Users/lucianateixeira/Desktop/calendar.csv')


# In[6]:


#Importing the dataset # 4

#The dataset chosen for this project is the Dublin listings from OpenData 

airbnb_listing_1 = pd.read_csv('/Users/lucianateixeira/Desktop/airbnb-listings.csv', sep=";")


# In[7]:


#Importing the dataset # 5

#The dataset chosen for this project is Dublin listings from OpenData 

airbnb_listing_2 = pd.read_csv('/Users/lucianateixeira/Desktop/airbnb-opendata.csv', sep=";")


# In[8]:


#Importing the dataset # 6

#The dataset chosen for this project is Dublin rate from Airbtics 
rate = pd.read_csv('/Users/lucianateixeira/Desktop/rate.csv')

#this dataset is a sample, however it contains a lot of information. I needed to fix it as it didn't have the columns set right.


# Some of the datasets are for trends and whether or not features influence reviews that influence occupancy that generates higher ROI. 
# The dataset for ML are Listings, REview and Rate (it is a sample data of Dublin) to identify the occupancy rate in the future. 

# **Dataset 1: Listings**
# 

# Data Understanding & Cleaning

# In[9]:


listing.info()


# In[10]:


# Drop the data that are not of interest and/or causing privacy issues
listing.drop(['id','host_name','last_review'], axis=1, inplace=True)
# Visualize the first 5 rows
listing.head(100)


# In[11]:


listing.isnull().sum()


# In[12]:


listing.info()


# In[13]:


listing.drop(['neighbourhood_group', 'license'], axis=1, inplace=True)
#listing.drop(['neighbourhood_group', 'license', 'name'], axis=1, inplace=True)


# **Data Visualisation - EDA Listing**

# In[14]:


# Visualizing the distribution for every "feature"
listing.hist(edgecolor="black", linewidth=1.2, figsize=(30, 30));


# In[15]:


plt.figure(figsize=(30, 30))
sns.pairplot(listing, height=3, diag_kind="hist")

#latitude and longitude have a normal distribution, most of the hosts are concetrated in specific area.
#reviews_per_month has a lot of outliers, because of the missing values filled by mean() and mode()
#price most the host has a price under $1000


# In[16]:


# Another way to visualize the data is to use FacetGrid to plot multiple kedplots on one plot

fig = sns.FacetGrid(listing, hue="neighbourhood", aspect=4, height=10)
fig.map(sns.kdeplot, 'host_id', shade=True)
oldest = listing['host_id'].max()
fig.set(xlim=(0, oldest))
sns.set(font_scale=5)
fig.add_legend()


# In[17]:


sns.set(font_scale=1.5)
sns.catplot("room_type", data=listing, kind="count", height=8)


# In[18]:


# Another way to visualize the data is to use FacetGrid to plot multiple kedplots on one plot

fig = sns.FacetGrid(listing, hue="room_type", aspect=4, height=10)
fig.map(sns.kdeplot, 'host_id', shade=True)
oldest = listing['host_id'].max()
fig.set(xlim=(0, oldest))
sns.set(font_scale=5)
fig.add_legend()


# In[19]:


sns.set(font_scale=1.5)
plt.figure(figsize=(12, 8))
listing.host_id.hist(bins=100)


# In[20]:


data = listing.neighbourhood.value_counts()[:10]
plt.figure(figsize=(12, 8))
x = list(data.index)
y = list(data.values)
x.reverse()
y.reverse()

plt.title("Most Popular Neighbourhood")
plt.ylabel("Neighbourhood Area")
plt.xlabel("Number of guest Who host in this Area")

plt.barh(x, y)


# In[21]:


listing['neighbourhood'].value_counts()

#Dublin City has 5334 properties listed out of 6977 in total
#Dun Laoghaire has 742
#Fingal has 616 and South Dublin represents the lower portion with only 285 properties listed.


# In[22]:


#Check property availability per neighbourhood
plt.figure(figsize=(14,6));
listing.groupby(['neighbourhood'])['availability_365'].mean().sort_values(ascending=False).plot(kind='bar');
plt.title('Availability Percentage as per Neighbourhood');
plt.xlabel('neighbourhood');
plt.ylabel('availability_365 %');


# In[23]:


listing['number_of_reviews'].value_counts()

#1450 out of 6977 Properties have 0 reviews. 
#653 properties have at least 1 review.
#463 properties have more than 1 review.


# In[24]:


plt.figure(figsize=(12, 8))
plt.scatter(listing.longitude, listing.latitude, c=listing.availability_365, cmap='summer', edgecolor='black', linewidth=1, alpha=0.75)

cbar = plt.colorbar()
cbar.set_label('availability_365')


plt.figure(figsize=(12, 8))
plt.scatter(listing.longitude, listing.latitude, c=listing.price, cmap='summer', edgecolor='black', linewidth=1, alpha=0.75)

cbar = plt.colorbar()
cbar.set_label('Price $')


# In[25]:


print(f"Average of price per night : ${listing.price.mean():.2f}")
print(f"Maximum price per night : ${listing.price.max()}")
print(f"Minimum price per night : ${listing.price.min()}")


# In[26]:


# group the listings by neighbourood and get the average price
listing.groupby('neighbourhood')['price'].mean().sort_values(ascending=False)


# In[27]:


(listing.groupby('neighbourhood')['price'].mean().sort_values(ascending=False)).plot(kind="bar", figsize=(16,8));
plt.title("Average price across Neighbourhoods");
plt.xlabel('Neighbourhood');
plt.ylabel('Average Price');

#Entire homes or apartments seem to be the most expensive type of properties in Ireland,
#followed by Hotel rooms, private rooms and shared rooms (hostels).


# In[28]:


#really expensive house

listing[listing.price == 1173721]


# In[29]:


#really cheap houses

listing[listing.price <10]


# In[30]:


#Limit price to 2000
ax = sns.boxplot( y="price", data=listing)
ax.set_ylim([0, 2000])


# In[31]:


# Computing IQR
q1 = listing['price'].quantile(0.25)
q3 = listing['price'].quantile(0.75)
iqr = q3 - q1
print(iqr)

print(q3+(1.5*iqr))

outliers= listing.loc[listing['price'] > q3+(1.5*iqr)]


# In[32]:


listing_without_outliers = listing.loc[listing['price'] <= q3+(1.5*iqr)]

#Boxplot
ax = sns.boxplot( y="price", data=listing_without_outliers)


# In[33]:


sub_1=listing_without_outliers[listing_without_outliers.price < 500]

import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')

vis_1=sub_1.plot(kind='scatter', x='longitude', y='latitude', label='neighbourhood', c='price',
                  cmap=plt.get_cmap('jet'), colorbar=True, alpha=0.4, figsize=(10,8))
vis_1.legend()


# In[34]:


# correlation matrix
sns.set(font_scale=3)
plt.figure(figsize=(30, 20))
sns.heatmap(listing.corr(), annot=True)


# In[35]:


plt.figure(figsize=(30, 30))
sns.set(font_scale=1.5)
i = 1
for column in listing.columns:
    if listing[column].dtype == "float64" or listing[column].dtype == "int64":
        plt.subplot(3, 3, i)
        listing.corr()[column].sort_values().plot(kind="barh")
        i += 1


# In[39]:


top_host= listing.host_id.value_counts().head(10)
top_host

top_host=pd.DataFrame(top_host)
top_host.reset_index(inplace=True)
top_host.rename(columns={'index':'Host_ID', 'host_id':'P_Count'}, inplace=True)
top_host


# In[40]:


top_host_check= listing.calculated_host_listings_count.max()
top_host_check


# In[41]:


barchart_host=sns.barplot(x="Host_ID", y="P_Count", data=top_host,
                 palette='Blues_d')
barchart_host.set_title('Hosts with the most listings in Dublin')
barchart_host.set_ylabel('Count of listings')
barchart_host.set_xlabel('Host IDs')
barchart_host.set_xticklabels(barchart_host.get_xticklabels(), rotation=45)


# In[42]:


listing['availability_365'].value_counts()

#4237 properties have 0 days available.
#Only 2740 properties out of 6977 had more than 1 day available for rental at the time this data was collected. 
#186 properties are available for at least 358 days a year.


# In[43]:


#let's what we can do with our given longtitude and latitude columns

#let's see how scatterplot will come out 
scatter_avail=listing.plot(kind='scatter', x='longitude', y='latitude', label='availability_365', c='price',
                  cmap=plt.get_cmap('jet'), colorbar=True, alpha=0.4, figsize=(10,8))
scatter_avail.legend()


# In[44]:


listing.fillna({'reviews_per_month':0}, inplace=True)


# In[45]:


(listing[['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
       'calculated_host_listings_count', 'availability_365']]
 .describe())


# In[46]:


listing = listing.loc[listing['price'] > 0]


# In[47]:


listing.describe()


# In[48]:


listing.head()


# In[49]:


# Recode data as categorical
listing_encoded = listing.copy()
listing_encoded['minimum_nights'] = pd.qcut(listing['minimum_nights'], q=2, labels=["minimum_nights_low", "minimum_nights_high"])
listing_encoded['number_of_reviews'] = pd.qcut(listing['number_of_reviews'], q=3, labels=["number_of_reviews_low", "minimum_nights_medium", "number_of_reviews_high"])
listing_encoded['reviews_per_month'] = pd.qcut(listing['reviews_per_month'], q=2, labels=["reviews_per_month_low", "reviews_per_month_high"])
listing_encoded['calculated_host_listings_count'] = pd.cut(listing['calculated_host_listings_count'], 
                                                bins=[0, 2, 327],
                                                labels=["calculated_host_listings_count_low", "calculated_host_listings_count_high"])

#NEEDS TO BE FIXED
#listing_encoded['availability_365'] = pd.qcut(listing['availability_365'], q=2, labels=["availability_low", "availability_high"])




# In[50]:


listing_encoded.isnull().sum()


# In[51]:


sns.set_palette("muted")
from pylab import *
f, ax = plt.subplots(figsize=(8, 6))

subplot(2,3,1)
sns.distplot(listing['price'])

subplot(2,3,2)
sns.distplot(listing['minimum_nights'])

subplot(2,3,3)
sns.distplot(listing['number_of_reviews'])

subplot(2,3,4)
sns.distplot(listing['reviews_per_month'])

subplot(2,3,5)
sns.distplot(listing['calculated_host_listings_count'])

subplot(2,3,6)
sns.distplot(listing['availability_365'])

plt.tight_layout() # avoid overlap of plotsplt.draw()


# In[52]:


title = 'Properties per Neighbourhood'
sns.countplot(listing['neighbourhood'])
plt.title(title)
plt.ioff()


# In[53]:


title = 'Properties per Room Type'
sns.countplot(listing['room_type'])
plt.title(title)
plt.ioff()


# In[54]:


plt.figure(figsize=(20,10))
title = 'Correlation matrix of numerical variables'
sns.heatmap(listing.corr(), square=True, cmap='RdYlGn')
plt.title(title)
plt.ioff()


# In[55]:


corr = listing.corr()


# In[56]:


corr.style.background_gradient(cmap = 'coolwarm')

#Strong Correlations were found between:
#The number_of_reviews_ltm and reviews_per_month columns (0.75)
#The number_of_reviews and reviews_per_month (0.55)
#No further important correlations were found.


# In[57]:


title = 'Neighbourhood Location'
plt.figure(figsize=(10,6))
sns.scatterplot(listing.longitude,listing.latitude,hue=listing.neighbourhood).set_title(title)
plt.ioff()

title = 'Room type location per Neighbourhood'
plt.figure(figsize=(10,6))
sns.scatterplot(listing.longitude,listing.latitude,hue=listing.room_type).set_title(title)
plt.ioff()


# In[58]:


listing.info()


# In[59]:


#initializing empty list where we are going to put our name strings
_names_=[]
#getting name strings from the column and appending it to the list
for name in listing.name:
    _names_.append(name)
#setting a function that will split those name strings into separate words   
def split_name(name):
    spl=str(name).split()
    return spl
#initializing empty list where we are going to have words counted
_names_for_count_=[]
#getting name string from our list and using split function, later appending to list above
for x in _names_:
    for word in split_name(x):
        word=word.lower()
        _names_for_count_.append(word)


# In[60]:


#we are going to use counter
from collections import Counter
#let's see top 25 used words by host to name their listing
_top_25_w=Counter(_names_for_count_).most_common()
_top_25_w=_top_25_w[0:25]


# In[61]:


#now let's put our findings in dataframe for further visualizations
sub_w=pd.DataFrame(_top_25_w)
sub_w.rename(columns={0:'Words', 1:'Count'}, inplace=True)


# In[62]:


viz_5=sns.barplot(x='Words', y='Count', data=sub_w)
viz_5.set_title('Counts of the top 25 used words for listing names')
viz_5.set_ylabel('Count of words')
viz_5.set_xlabel('Words')
viz_5.set_xticklabels(viz_5.get_xticklabels(), rotation=90)


# In[63]:


sub_1=listing.loc[listing['neighbourhood'] == 'Dn Laoghaire-Tathdown']
price_sub1=sub_1[['price']]
#Manhattan
sub_2=listing.loc[listing['neighbourhood'] == 'South Dublin']
price_sub2=sub_2[['price']]
#Queens
sub_3=listing.loc[listing['neighbourhood'] == 'Dublin City']
price_sub3=sub_3[['price']]
#Staten Island
sub_4=listing.loc[listing['neighbourhood'] == 'Fingal']
price_sub4=sub_4[['price']]

#putting all the prices' dfs in the list
price_list_by_n=[price_sub1, price_sub2, price_sub3, price_sub4]


# In[64]:


p_l_b_n_2=[]
#creating list with known values in neighbourhood_group column
nei_list=['Dn Laoghaire-Tathdown', 'South Dublin', 'Dublin City', 'Fingal']
#creating a for loop to get statistics for price ranges and append it to our empty list
for x in price_list_by_n:
    i=x.describe(percentiles=[.25, .50, .75])
    i=i.iloc[3:]
    i.reset_index(inplace=True)
    i.rename(columns={'index':'Stats'}, inplace=True)
    p_l_b_n_2.append(i)
#changing names of the price column to the area name for easier reading of the table    
p_l_b_n_2[0].rename(columns={'price':nei_list[0]}, inplace=True)
p_l_b_n_2[1].rename(columns={'price':nei_list[1]}, inplace=True)
p_l_b_n_2[2].rename(columns={'price':nei_list[2]}, inplace=True)
p_l_b_n_2[3].rename(columns={'price':nei_list[3]}, inplace=True)

#finilizing our dataframe for final view    
stat_df=p_l_b_n_2
stat_df=[df.set_index('Stats') for df in stat_df]
stat_df=stat_df[0].join(stat_df[1:])
stat_df


# In[65]:


#creating a sub-dataframe with no extreme values / less than 500
sub_6=listing[listing.price < 500]
#using violinplot to showcase density and distribtuion of prices 
viz_2=sns.violinplot(data=sub_6, x='neighbourhood', y='price')
viz_2.set_title('Density and distribution of prices for each neighberhood')


# In[66]:


# Set up color blind friendly color palette
# The palette with grey:
#cbPalette = ["#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
# The palette with black:
#cbbPalette = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

# sns.palplot(sns.color_palette(cbPalette))
# sns.palplot(sns.color_palette(cbbPalette))

#sns.set_palette(cbPalette)
#sns.set_palette(cbbPalette)


# In[67]:


#View the top 10 properties most reviewed and get the average price per night
top_reviewed_listings=listing.nlargest(10,'number_of_reviews')
top_reviewed_listings


# In[68]:


price_avrg=top_reviewed_listings.price.mean()
print('Average price per night: {}'.format(price_avrg))


#These properties were the most reviewed from the dataset. 
#However, it is relevant to point out that they don't necessarily have the highest reviews. 
#In addition, another plot could be added in the future to check if there is a correlation between the highest reviews and the price per night. 
#90% of these properties are located in Dublin City. Which is another indication that Dublin City is indeed a relevant neighbourhood to consider when investing on a property or even when looking for accomodation.
#The average price per night of these properties is 77.70 euros


# In[69]:


listing.info()


# **Models and predictions**
# 
# Price is the desired prediction.

# In[70]:


#Linear regression 
X = listing['price'].values.reshape(-1,1)
y = listing['availability_365'].values.reshape(-1,1)


# In[71]:


#split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

regressor = LinearRegression()  
regressor.fit(X_train, y_train) #train the model


# In[72]:


#Intercept
print(regressor.intercept_)
#Slope
print(regressor.coef_)


# In[73]:


#Predict
y_pred = regressor.predict(X_test)


# In[74]:


#actual value and predicted value
dfLinReg = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
dfLinReg


# In[75]:


import sklearn.metrics as metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


#The above linear regression model is not ideal.
#Changes to test size and random state did not impact the results


# In[76]:


#Multiple Regression

#Convert to Numeric
# creating instance of labelencoder
labelencoder = LabelEncoder()# Assigning numerical values and storing in another column
listing['room_type_Cat'] = labelencoder.fit_transform(listing['room_type'])
listing['city_Cat'] = labelencoder.fit_transform(listing['neighbourhood'])
listing.head()


# In[77]:


#multiple regression

X_ = listing[['calculated_host_listings_count', 'room_type_Cat', 
          'room_type_Cat', 'city_Cat']] # multiple variable regression. 
Y = listing['price']
 
# with sklearn
regr = LinearRegression()
regr.fit(X_, Y)

print('Intercept: ', regr.intercept_)
print('Coefficients: ', regr.coef_)

print (regr)


# In[78]:


y_pred2 = regr.predict(X_)


# In[79]:


dfmult= pd.DataFrame({'Actual': Y, 'Predicted': y_pred2.flatten()})
dfmult


# In[80]:


print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y, y_pred2)))


# In[81]:


#comparison 
first20preds2=dfmult.head(20)
first20preds2.plot(kind='bar',figsize=(9,5))
plt.grid(which='major', linestyle='-', linewidth='0.3', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[82]:


#KNN
# create a list of conditions
conditions = [
    (listing['price'] < 100),
    (listing['price'] >= 100) & (listing['price'] < 250),
     (listing['price'] >= 250) & (listing['price'] < 600),
    (listing['price'] >= 600) ]

# create a list of the values we want to assign for each condition
values = ['economic', 'low-mid', 'high-mid','high']

# create a new column and use np.select to assign values to it using our lists as arguments
listing['price_range'] = np.select(conditions, values)


# In[83]:


listing['price_range'].value_counts()


# In[84]:


labelencoder = LabelEncoder()# Assigning numerical values and storing in another column
listing['price_rng_Cat'] = labelencoder.fit_transform(listing['price_range'])


# In[85]:


knn=listing[['minimum_nights','availability_365', 
         'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count',
         'room_type_Cat','city_Cat', 'price', 'price_rng_Cat']]


# In[86]:


knn.corr().style.background_gradient(cmap='magma')


# In[87]:


#conda install -c conda-forge scikit-learn


# In[88]:


#K-nearest neighbors (KNN)
##Best performing model

from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3)

X1=knn.iloc[:, :-1].values
Y1=knn['price_rng_Cat'].values

# Split into training and test  
X_train, X_test, y_train, y_test = train_test_split( 
             X1, Y1, test_size = 0.4, random_state=1) 

#standardize data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#fit the model
neigh.fit(X_train, y_train)


# In[89]:


# Predicted class
y_pred3=neigh.predict(X_test)


# In[90]:


KNNmod = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred3.flatten()})
KNNmod


# In[91]:


# Calculate the accuracy of the model 
print(neigh.score(X_test, y_test)) 

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred3)))


# In[92]:


#comparison 
first20preds3=KNNmod.head(20)
c2='darkkhaki', 'dimgray'
first20preds3.plot(kind='barh',figsize=(9,6), color=c2)
plt.grid(which='major', linestyle='-', linewidth='0.3', color='orange')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# **Dataset 2: Review**

# Data Understanding and Cleaning

# In[ ]:


review.head()


# In[ ]:


#Checking and removing if there are null values in the data.
review.isnull().sum()


# In[ ]:


#Filling up null values with No review for generate word cloud.
review.fillna({'comments': 'Not available'}, inplace = True)


# In[ ]:


review.isnull().sum()


# In[ ]:


#Dropping columns that are not needed for the future model
review.drop(['id', 'listing_id','date','reviewer_id','reviewer_name'], axis=1, inplace=True)


# In[ ]:


review.info()


# **Data Visualisation -EDA**

# In[ ]:





# **Sentiment Analysis**

# Sentiment analysis is important to understand how much of impact the reviews are for a businness but more for this dataset it is important to discover positive or negative reviews of locations in Dublin.
# 
# It will use sentiment analysis from 1 to 5

# In[ ]:


get_ipython().system('pip install torch torchvision torchaudio')


# In[ ]:


conda install pytorch torchvision torchaudio -c pytorch


# In[ ]:


get_ipython().system('pip install transformers requests beautifulsoup4')


# In[ ]:


pip install transformers


# In[ ]:


pip install torchvision 


# In[ ]:


conda install statsmodels


# In[ ]:


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup

from statsmodels.stats.power import TTestIndPower


# In[ ]:


#Instantiating the model

tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')


# In[ ]:


# Encoding and calculating sentiment

#A test of a negative review
tokens = tokenizer.encode('I did not like my stay, could be better', return_tensors='pt')


# In[ ]:


result = model(tokens)


# In[ ]:


#Below is the result logits inside torch plus one to display the sentiment
int(torch.argmax(result.logits))+1


# ### Sample size: 
# As this dataframe contains over 221.000 rows to create an sentiment for the future will take a while,so using a power analysis to discover the sample size will allow us to the best for our future model.

# In[ ]:


#Start of power analysis
power_analysis = TTestIndPower()


# In[ ]:


#Calculating the sample size
sample_size = power_analysis.solve_power(effect_size = 0.2, alpha = 0.05, power = 0.8, alternative = 'two-sided')


# In[ ]:


#Results containing the sample
print('The sample size needed is: ', round(sample_size))


# In[ ]:


#Selecting the first 500 rows of the dataframe to keep

review2 = review.head(500)


# In[ ]:


#conda install -c conda-forge spacy


# In[ ]:


#pip install -U pip setuptools wheel


# In[ ]:


#pip install -U spacy


# In[ ]:


#pip instal -U spacy


# In[ ]:


#conda install tensorflow 


# In[ ]:


#Scoring sentiment from dataframe
import re
import spacy
import nltk
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from wordcloud import WordCloud,STOPWORDS
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


# In[ ]:


#For creating the scoring the comments requires cleaning.
review2


# In[ ]:


review2['comments'].iloc[0] 


# In[ ]:


#Creating a method for scoring the sentiment

def sentiment_score(comments):
    tokens = tokenizer.encode(comments, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))+1


# In[ ]:


#Diplay score sentiment of first row

sentiment_score(review2['comments'].iloc[1] )


# In[ ]:


#Applying the scores to the dataframe.

review2['sentiment'] = review2['comments'].apply(lambda x: sentiment_score(x[:512]))


# In[ ]:


review2


# In[ ]:


#Tokenization for prediction with sentiment
#Cleaning method for tokenization

stop_words = set(stopwords.words("english")) 
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text

review2['Comments_processed'] = review2.comments.apply(lambda x: clean_text(x))


# In[ ]:


#Show comments processed

review2.head()


# In[ ]:


#Find the mean of all the comments
review2.Comments_processed.apply(lambda x: len(x.split(" "))).mean()


# In[ ]:


#Creating model for prediction


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_features = 6000
tokenizer = Tokenizer()
tokenizer.fit_on_texts(review2['Comments_processed'])
list_tokenized_train = tokenizer.texts_to_sequences(review2['Comments_processed'])

maxlen = 130
X_train = pad_sequences(list_tokenized_train, maxlen=maxlen)
y = review2['sentiment']

embed_size = 128
model = Sequential()
model.add(Embedding(max_features, embed_size))
model.add(Bidirectional(LSTM(32, return_sequences = True)))
model.add(Dense(20, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 100
epochs = 3
model.fit(X_train,y, batch_size=batch_size, epochs=epochs, validation_split=0.2)


# In[ ]:


#Word Cloud
#Visualization of most use words


def wordCloud_generator(review2, title=None):
    wordcloud = WordCloud(width = 800, height = 800,
                          background_color ='white',
                          min_font_size = 10
                         ).generate(" ".join(review2.values))                      
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud, interpolation='bilinear') 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.title(title,fontsize=30)
    plt.show()
wordCloud_generator(review2['comments'], title="Top words in reviews")


# **Dataset 3: Calendar**

# In[ ]:


calendar.info()


# In[ ]:


#change the date column to a datetime 
calendar['date'] = pd.to_datetime(calendar.date)
calendar.info(verbose=True, null_counts=True)


# In[ ]:


#missing values in Calender data

plt.figure(figsize=(10,6));
null_price = calendar.isnull().sum()
(null_price/calendar.shape[0]).plot(kind='bar');


# In[ ]:


#removing null dates
calendar.dropna(axis=0,subset=['price'],inplace=True)


# In[ ]:


#data prep/wrangling

#Calender dataset

# change the date column to a datetime 
calendar['date'] = pd.to_datetime(calendar.date)
calendar.info(verbose=True, null_counts=True)


# In[ ]:


# drop the missing values in price and drop 
calendar= calendar.dropna(subset=['price'], axis = 0)
calendar.info(verbose=True, null_counts=True)


# In[ ]:


#function to clean and convert str columns to numeric

def str_to_num(df,column):
    df[column] = pd.to_numeric(df[column].apply(lambda x : str(x).replace('$','').replace(",",'')),errors='coerce')
    return df


# In[ ]:


columns = ['price','adjusted_price']

for col in columns:
    calendar = str_to_num(calendar,col)

calendar[columns][:2]


# In[ ]:


# to clean the price and convert it into a float 
calendar = str_to_num(calendar,'price')
calendar.info(verbose=True, null_counts=True)


# In[ ]:


calendar.describe()


# In[ ]:


# add month and year column to the calender dataset
calendar['month'], calendar['year'] = calendar.date.dt.month, calendar.date.dt.year
calendar.info(verbose=True, null_counts=True)


# In[ ]:


#effect on booking prices at different times of the year

calendar.available.value_counts()


# In[ ]:


#We can observe here that there is a price hike in mid of the year and prices are lowest at the start of the year.
import plotly.offline as pyoff

price = pd.DataFrame(calendar.groupby(['month','available']).mean()['price'].reset_index())

data = [
    go.Scatter(
        x = price['month'],
        y = price.price,
        name = 'Price'
    )
]

layout = go.Layout(
    title = 'Booking prices as per months',
    xaxis = dict(title='Months'),
    yaxis = dict(title= '$ Price'),
    showlegend=True,
    
)
fig = go.Figure(data=data,layout=layout)

pyoff.iplot(fig)


# In[ ]:


#as per the dataset all listings are available

available_count_daily = calendar.groupby('date').count()[['price']]
available_count_daily = available_count_daily.rename({"price":"total_available_houses"},axis='columns')

average_price_daily = calendar.groupby('date').mean()[['price']]
# change column name
average_price_daily = average_price_daily.rename({"price":"average_prices"},axis='columns')


# In[ ]:


# plot total available houses and average prices in one figure
f, ax = plt.subplots(figsize=(15, 6))
plt1 = sns.lineplot(x = available_count_daily.index,y = 'total_available_houses', 
                  data = available_count_daily,color="red",legend=False,label='No. of houses available')

ax2 = ax.twinx()
plt2 = sns.lineplot(x = average_price_daily.index,y = 'average_prices',
             data=average_price_daily,color = 'black', ax=ax2,linestyle=':', legend=False,label='Daily prices')
ax.set_title('Comparing the daily availability of airbnb listing with the daily listing prices');
ax2.legend();
ax.legend();


# In[ ]:


features = calendar.columns[:-1].tolist()
print(calendar.shape)


# In[ ]:


sns.set(style="whitegrid", font_scale=1.2)
plt.subplots(figsize = (8,8))
sns.countplot('available',data=calendar).set_title('Available for f and t')


# In[ ]:


title = 'Availabilty for t and f'
plt.figure(figsize=(10,6))
sns.scatterplot(calendar.maximum_nights,calendar.minimum_nights,hue=calendar.available).set_title(title)
plt.ioff()


# In[ ]:


(calendar.groupby('minimum_nights')['price'].mean().sort_values(ascending=False)[:20]).plot(kind="bar", figsize=(16,8));
plt.title("Average Listing price");
plt.xlabel('Average Price $');
plt.ylabel('minimum nights');
plt.xticks(rotation=60);


# In[ ]:


(calendar.groupby('maximum_nights')['price'].mean().sort_values(ascending=False)[:50]).plot(kind="bar", figsize=(16,8));
plt.title("Average Listing price");
plt.xlabel('Average Price $');
plt.ylabel('Maximum nights');
plt.xticks(rotation=50);


# **Dataset 4 and 5: Airbnb Listing 1 and 2**
# 
# These two datasets were chosen to show visualisation and offer marketing insights and trends. 
# They will be merged together. 

# In[ ]:


airbnb_listing_1.info()


# In[ ]:


airbnb_listing_1.head(10)


# In[ ]:


airbnb_listing_2.info()


# In[ ]:


airbnb_listing_2.head(10)


# In[ ]:


#checking the size of the dataset, rows and columns
airbnb_listing_1.shape


# In[ ]:


#checking the size of the dataset, rows and columns
airbnb_listing_2.shape


# In[ ]:


#A resultant dataframe which has the rows from the target dataframe and a new row appended.

airbnb1 = pd.DataFrame(airbnb_listing_1)

airbnb1

airbnb2 = pd.DataFrame(airbnb_listing_2)

airbnb2


# In[ ]:


#The append() method in python adds a single item to the existing list. 
#It doesn't return a new list of items but will modify the original list by adding the item to the end of the list. 
#After executing the method append on the list the size of the list increases by one.

#ignore index makes it not repeat the index. 


airbnb_final = airbnb1.append(airbnb2, ignore_index = True)

airbnb_final


# In[ ]:


#confirming the rows have been added to the the first dataset, now called airbnb_final. 

airbnb_final.shape


# In[ ]:


airbnb_final.isnull().sum()


# In[ ]:


airbnb_final.columns


# In[ ]:


airbnb_final[airbnb_final.duplicated()]


# In[ ]:


#Althought the values of the row of one the dataset were duplicated, we have decided to leave it here.
#The datset were soureced from different websites and we left to showcase the skills of identifying this i n large datasets. 

#removing the duplicated columns

#Keeping the last duplicate value, so that it removes just the one that is duplicated. 
airbnb_final.drop_duplicates(keep='last',inplace=True)

#airbnb_final = airbnb_final.loc[:,~airbnb_final.columns.duplicated()]


# In[ ]:


#checking the shape of the dataset
airbnb_final.shape


# In[ ]:


#dropped 36 columns

#some of the rationale include privacy concerns: photo URL, addresses and PII. 

airbnb_clean = airbnb_final.drop(columns=['Listing Url', 'Scrape ID', 'Last Scraped', 'Name', 'Summary',
       'Space', 'Description', 'Neighborhood Overview',
       'Notes', 'Transit', 'Access', 'Interaction', 'House Rules',
       'Thumbnail Url', 'Medium Url', 'Picture Url', 'XL Picture Url',
       'Host URL', 'Host Name', 'Calendar Updated', 'Calendar last Scraped',
       'Host About','Host Thumbnail Url', 'Host Picture Url', 'Host Verifications', 'Street',
       'Has Availability', 'Availability 30', 'Availability 60',
       'Availability 90', 'First Review', 'Last Review','License', 'Jurisdiction Names',
       'Cancellation Policy', 'Calculated host listings count'])

airbnb_clean.head()


# In[ ]:


airbnb_clean.isnull().sum()


# In[ ]:


airbnb_clean.info()


# In[ ]:


#replacing all NaN values with 0 for categorical data
airbnb_clean.fillna({'Host Location':0}, inplace=True)
airbnb_clean.fillna({'Host Response Time':0}, inplace=True)
airbnb_clean.fillna({'Host Acceptance Rate':0}, inplace=True)
airbnb_clean.fillna({'Host Neighbourhood':0}, inplace=True)
airbnb_clean.fillna({'Neighbourhood':0}, inplace=True)
airbnb_clean.fillna({'Neighbourhood Group Cleansed':0}, inplace=True)
airbnb_clean.fillna({'City':0}, inplace=True)
airbnb_clean.fillna({'State':0}, inplace=True)
airbnb_clean.fillna({'Zipcode':0}, inplace=True)
airbnb_clean.fillna({'Market':0}, inplace=True)
airbnb_clean.fillna({'Amenities':0}, inplace=True)
airbnb_clean.fillna({'Features':0}, inplace=True)


# In[ ]:


airbnb_clean.isnull().sum()


# In[ ]:


#The first chosen technique is one of the possible ways of dealing with null values by using median 
#to impute the missing values.
#Median is better than mean, because mean is influenced by the outliers. 

missing_col = ['Host Response Rate']
for i in missing_col:
 airbnb_clean.loc[airbnb_clean.loc[:,i].isnull(),i]=airbnb_clean.loc[:,i].median()

missing_col = ['Bathrooms']
for i in missing_col:
 airbnb_clean.loc[airbnb_clean.loc[:,i].isnull(),i]=airbnb_clean.loc[:,i].median()

missing_col = ['Bedrooms']
for i in missing_col:
 airbnb_clean.loc[airbnb_clean.loc[:,i].isnull(),i]=airbnb_clean.loc[:,i].median()

missing_col = ['Beds']
for i in missing_col:
 airbnb_clean.loc[airbnb_clean.loc[:,i].isnull(),i]=airbnb_clean.loc[:,i].median()

missing_col = ['Square Feet']
for i in missing_col:
 airbnb_clean.loc[airbnb_clean.loc[:,i].isnull(),i]=airbnb_clean.loc[:,i].median()

missing_col = ['Price']
for i in missing_col:
 airbnb_clean.loc[airbnb_clean.loc[:,i].isnull(),i]=airbnb_clean.loc[:,i].median()

missing_col = ['Weekly Price']
for i in missing_col:
 airbnb_clean.loc[airbnb_clean.loc[:,i].isnull(),i]=airbnb_clean.loc[:,i].median()

missing_col = ['Monthly Price']
for i in missing_col:
 airbnb_clean.loc[airbnb_clean.loc[:,i].isnull(),i]=airbnb_clean.loc[:,i].median()

missing_col = ['Security Deposit']
for i in missing_col:
 airbnb_clean.loc[airbnb_clean.loc[:,i].isnull(),i]=airbnb_clean.loc[:,i].median()

missing_col = ['Cleaning Fee']
for i in missing_col:
 airbnb_clean.loc[airbnb_clean.loc[:,i].isnull(),i]=airbnb_clean.loc[:,i].median()

missing_col = ['Review Scores Rating']
for i in missing_col:
 airbnb_clean.loc[airbnb_clean.loc[:,i].isnull(),i]=airbnb_clean.loc[:,i].median()

missing_col = ['Review Scores Accuracy']
for i in missing_col:
 airbnb_clean.loc[airbnb_clean.loc[:,i].isnull(),i]=airbnb_clean.loc[:,i].median()

missing_col = ['Review Scores Cleanliness']
for i in missing_col:
 airbnb_clean.loc[airbnb_clean.loc[:,i].isnull(),i]=airbnb_clean.loc[:,i].median()

missing_col = ['Review Scores Checkin']
for i in missing_col:
 airbnb_clean.loc[airbnb_clean.loc[:,i].isnull(),i]=airbnb_clean.loc[:,i].median()

missing_col = ['Review Scores Communication']
for i in missing_col:
 airbnb_clean.loc[airbnb_clean.loc[:,i].isnull(),i]=airbnb_clean.loc[:,i].median()


missing_col = ['Review Scores Location']
for i in missing_col:
 airbnb_clean.loc[airbnb_clean.loc[:,i].isnull(),i]=airbnb_clean.loc[:,i].median()

missing_col = ['Review Scores Value']
for i in missing_col:
 airbnb_clean.loc[airbnb_clean.loc[:,i].isnull(),i]=airbnb_clean.loc[:,i].median()

missing_col = ['Reviews per Month']
for i in missing_col:
 airbnb_clean.loc[airbnb_clean.loc[:,i].isnull(),i]=airbnb_clean.loc[:,i].median()


# In[ ]:


#Confirming the datset does not have any null values, they have been replaced by the median. 

airbnb_clean.isnull().sum()


# **Data Visualisation**

# In[ ]:


columns_location_categ = list(airbnb_clean[['Host Neighbourhood', 'Neighbourhood', 
                                        'Neighbourhood Cleansed','Neighbourhood Group Cleansed', 
                                        'City', 'State', 'Zipcode', 'Market','Smart Location', 
                                            'Country Code', 'Country', 'Latitude', 'Longitude', 'Geolocation'
                                           ]]) 


# In[ ]:


# Check the number of unique values in each column
for i in columns_location_categ: 
        num_unique=(airbnb_clean[i].nunique())
        num_isnull=(airbnb_clean[i].isnull().sum())
        print('Field name: {}, Number of null records: {}, Number of unique values: {}'.format(i, num_isnull, num_unique))
        print('+'*90)
        print('List of unique values:')
        print(airbnb_clean[i].unique())
        print(' '*150)
        print('-'*150)
        print(' '*150)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# **Dataset 6: Rate**

# In[ ]:


rate.head()


# In[ ]:


#Based on the information on the dataset itself, I added these columns names. 
rate.columns = [
    "Listing URL",
    "Property Type",
    "Latitude",
    "Longitude",
    "Star Rating",
    "Number of Active Days",
    "Bedrooms",
    "Has pool",
    "Cleaning Fee",
    "Extra Guest Fee",
    "Daily Rate (2020-11)",
    "Daily Rate (2020-12)",
    "Daily Rate (2021-01)",
    "Daily Rate (2021-02)",
    "Daily Rate (2021-03)",
    "Daily Rate (2021-04)",
    "Daily Rate (2021-05)",
    "Daily Rate (2021-06)",
    "Daily Rate (2021-07)",
    "Daily Rate (2021-08)",
    "Daily Rate (2021-09)",
    "Daily Rate (2021-10)",
    "Occupancy Rate (2020-11)",
    "Occupancy Rate (2020-12)",
    "Occupancy Rate (2021-01)",
    "Occupancy Rate (2021-02)",
    "Occupancy Rate (2021-03)",
    "Occupancy Rate (2021-04)",
    "Occupancy Rate (2021-05)",
    "Occupancy Rate (2021-06)",
    "Occupancy Rate (2021-07)",
    "Occupancy Rate (2021-08)",
    "Occupancy Rate (2021-09)",
    "Occupancy Rate (2021-10)",
    "Revenue (2020-11)",
    "Revenue (2020-12)",
    "Revenue (2021-01)",
    "Revenue (2021-02)",
    "Revenue (2021-03)",
    "Revenue (2021-04)",
    "Revenue (2021-05)",
    "Revenue (2021-06)",
    "Revenue (2021-07)",
    "Revenue (2021-08)",
    "Revenue (2021-09)",
    "Revenue (2021-10)",
]


# In[ ]:


rate.head()


# In[ ]:


#dropping the two first rows.
rate.drop(index=rate.index[0:2], 
        axis=0, 
        inplace=True)


# In[ ]:


rate.head()


# In[ ]:


#replacing all NaN values with 0
rate.fillna({'Cleaning Fee':0}, inplace=True)
rate.fillna({'Extra Guest Fee':0}, inplace=True)


# In[ ]:





# In[ ]:


rate.isnull().sum()


# In[ ]:


#examine the dataset
(rate[['Star Rating', 'Number of Active Days', 'Daily Rate (2020-12)', 'Occupancy Rate (2020-12)',
       'Revenue (2020-12)']]
 .describe())


# In[ ]:


#need to be encoded to integer

rate.info()


# In[ ]:


#separating the dataset into categorical and numeric

#later to change the object type to float64

categorical_col = ['Listing URL','Property Type', 'Has pool']

numeric_col = ['Latitude', 
               'Longitude', 
               'Star Rating', 
               'Number of Active Days', 
               'Bedrooms', 
               'Cleaning Fee', 
               'Extra Guest Fee',            
               'Daily Rate (2020-11)',         
               'Daily Rate (2020-12)',
               'Daily Rate (2021-01)',
               'Daily Rate (2021-02)',        
               'Daily Rate (2021-03)',       
                'Daily Rate (2021-04)',        
                'Daily Rate (2021-05)',        
                'Daily Rate (2021-06)',        
                'Daily Rate (2021-07)',        
                'Daily Rate (2021-08)',        
                'Daily Rate (2021-09)',        
                'Daily Rate (2021-10)',        
                'Occupancy Rate (2020-11)',    
                'Occupancy Rate (2020-12)',   
                'Occupancy Rate (2021-01)',    
                'Occupancy Rate (2021-02)',    
                'Occupancy Rate (2021-03)',    
                'Occupancy Rate (2021-04)',    
                'Occupancy Rate (2021-05)',    
                'Occupancy Rate (2021-06)',    
                'Occupancy Rate (2021-07)',    
                'Occupancy Rate (2021-08)',    
                'Occupancy Rate (2021-09)',    
                'Occupancy Rate (2021-10)',   
                'Revenue (2020-11)',           
                'Revenue (2020-12)',         
                'Revenue (2021-01)',           
                'Revenue (2021-02)',           
                'Revenue (2021-03)',           
                'Revenue (2021-04)',           
                'Revenue (2021-05)',           
                'Revenue (2021-06)',           
                'Revenue (2021-07)',           
                'Revenue (2021-08)',           
                'Revenue (2021-09)',           
                'Revenue (2021-10)',
              ]


# In[ ]:


print(numeric_col)


# In[ ]:


rate[numeric_col] = rate[numeric_col].apply(pd.to_numeric, errors='coerce')


# In[ ]:


rate.info()


# In[ ]:


#NOT NEEDED - things I tried.
#encoding the dataset so that I can add columns for the rates values. 
#rate['Cleaning Fee'] = rate['Cleaning Fee'].astype(float)


# In[ ]:


#rate.drop(columns=['Listing URL', 'Property Type', 'Latitude', 'Longitude', 'Has pool'])


# In[ ]:


rate.head()


# In[ ]:


#rate = rate.drop(['Daily Rate (2021-01)','Daily Rate (2021-02)', 'Daily Rate (2021-03)','Daily Rate (2021-04)', 'Daily Rate (2021-05)','Daily Rate (2021-06)', 'Daily Rate (2021-07)','Daily Rate (2021-08)', 'Daily Rate (2021-09)','Daily Rate (2021-10)', 'Daily Rate (2020-11)','Daily Rate (2020-12)','Revenue (2020-11)','Revenue (2020-12)','Revenue (2021-01)','Revenue (2021-02)','Revenue (2021-03)','Revenue (2021-04)','Revenue (2021-05)','Revenue (2021-06)','Revenue (2021-07)','Revenue (2021-08)','Revenue (2021-09)','Revenue (2021-10)','Occupancy Rate (2020-11)','Occupancy Rate (2020-12)','Occupancy Rate (2021-01)','Occupancy Rate (2021-02)','Occupancy Rate (2021-03)','Occupancy Rate (2021-04)','Occupancy Rate (2021-05)','Occupancy Rate (2021-06)','Occupancy Rate (2021-07)','Occupancy Rate (2021-08)','Occupancy Rate (2021-09)','Occupancy Rate (2021-10)'], axis=1)


# In[ ]:


#del rate['Daily Rate (2021-01)','Daily Rate (2021-02)', 'Daily Rate (2021-03)','Daily Rate (2021-04)', 'Daily Rate (2021-05)','Daily Rate (2021-06)', 'Daily Rate (2021-07)','Daily Rate (2021-08)', 'Daily Rate (2021-09)','Daily Rate (2021-10)', 'Daily Rate (2020-11)','Daily Rate (2020-12)','Revenue (2020-11)','Revenue (2020-12)','Revenue (2021-01)','Revenue (2021-02)','Revenue (2021-03)','Revenue (2021-04)',          'Revenue (2021-05)','Revenue (2021-06)','Revenue (2021-07)','Revenue (2021-08)','Revenue (2021-09)','Revenue (2021-10)','Occupancy Rate (2020-11)','Occupancy Rate (2020-12)','Occupancy Rate (2021-01)','Occupancy Rate (2021-02)','Occupancy Rate (2021-03)','Occupancy Rate (2021-04)','Occupancy Rate (2021-05)','Occupancy Rate (2021-06)','Occupancy Rate (2021-07)','Occupancy Rate (2021-08)','Occupancy Rate (2021-09)','Occupancy Rate (2021-10)']


# In[ ]:


#Check how many properties have a pool
rate['Has pool'].value_counts()


# In[ ]:


rate['Number of Active Days'] = rate['Number of Active Days'].astype('int') 


# In[ ]:


rate['Cleaning Fee'] = rate['Cleaning Fee'].astype('int') 


# In[ ]:


rate['ExtravGuest Fee'] = rate['Extra Guest Fee'].astype('float') 


# In[ ]:


rate['Revenue (2020-11)'] = rate['Revenue (2020-11)'].astype('int') 


# In[ ]:


rate['Revenue (2020-12)'] = rate['Revenue (2020-12)'].astype('int') 


# In[ ]:


rate['Revenue (2021-01)'] = rate['Revenue (2021-01)'].astype('int') 


# In[ ]:


rate['Revenue (2021-02)'] = rate['Revenue (2021-02)'].astype('int') 


# In[ ]:


rate['Revenue (2021-03)'] = rate['Revenue (2021-03)'].astype('int') 


# In[ ]:


rate['Revenue (2021-04)'] = rate['Revenue (2021-04)'].astype('int') 


# In[ ]:


rate['Revenue (2021-05)'] = rate['Revenue (2021-05)'].astype('int') 


# In[ ]:


rate['Revenue (2021-06)'] = rate['Revenue (2021-06)'].astype('int') 


# In[ ]:


rate['Revenue (2021-07)'] = rate['Revenue (2021-07)'].astype('int') 


# In[ ]:


rate['Revenue (2021-08)'] = rate['Revenue (2021-08)'].astype('int') 


# In[ ]:


rate['Revenue (2021-09)'] = rate['Revenue (2021-09)'].astype('int') 


# In[ ]:


rate['Revenue (2021-10)'] = rate['Revenue (2021-10)'].astype('int') 


# In[ ]:


#2020-11
rate['Occupancy Rate (2020-11)'].value_counts()


# In[ ]:


#2020-11
rate['Daily Rate (2020-11)'].value_counts()


# In[ ]:


#2020-11
rate['Revenue (2020-11)'].value_counts()


# In[ ]:


#2021-10
rate['Occupancy Rate (2021-10)'].value_counts()


# In[ ]:


#2021-10
rate['Daily Rate (2021-10)'].value_counts(ascending=False)


# In[ ]:


#2021-10
rate['Revenue (2021-10)'].value_counts(ascending=False)


# In[ ]:


#Pair Plot
#sns.pairplot(rate)
#plt.figure(figsize=(10,7))


# In[ ]:


plt.figure(figsize=(10,6))
rate.Bedrooms.hist();
plt.title('Bedroom distribution');
plt.xlabel('Number of Bedrooms');
plt.ylabel('Count');


# In[ ]:


rate['Cleaning Fee'].value_counts()


# In[ ]:


rate['Extra Guest Fee'].describe()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


rate.info()


# In[ ]:


#correlations - example 

rate.corr() # Whole correlation matrix
rate.corr()['revenue_mean'] # Check correlations with outcome only
sns.heatmap(rate.corr(), cmap="seismic", annot=True, vmin=-1, vmax=1);


# In[ ]:


#correlations - example 

rate.corr() # Whole correlation matrix
rate.corr()['daily_rate_mean'] # Check correlations with outcome only
sns.heatmap(rate.corr(), cmap="seismic", annot=True, vmin=-1, vmax=1);


# In[ ]:


#correlations - example 

rate.corr() # Whole correlation matrix
rate.corr()['occupancy_mean'] # Check correlations with outcome only
sns.heatmap(rate.corr(), cmap="seismic", annot=True, vmin=-1, vmax=1);


# In[ ]:


# Import packages
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import lars_path
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.metrics import r2_score
import scipy.stats as stats
# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


sns.distplot(rate['daily_rate_mean'], kde=True,);
fig = plt.figure()
res = stats.probplot(rate['daily_rate_mean'], plot=plt)
print("Skewness: %f" % rate['daily_rate_mean'].skew())
print("Kurtosis: %f" % rate['daily_rate_mean'].kurt())

#Need to review this as the dataset looks skewed 


# In[ ]:


rate.head()


# **Data Visualisation**

# In[ ]:





# **Predicting the Occupancy Rate**

# In[ ]:


rate


# In[ ]:


rate.isnull().sum()


# In[ ]:


#mean all the Revenue, Occupancy and Daily Rate

rate['daily_rate_mean'] = rate[['Daily Rate (2021-01)','Daily Rate (2021-02)', 'Daily Rate (2021-03)','Daily Rate (2021-04)', 'Daily Rate (2021-05)','Daily Rate (2021-06)', 'Daily Rate (2021-07)','Daily Rate (2021-08)', 'Daily Rate (2021-09)','Daily Rate (2021-10)', 'Daily Rate (2020-11)','Daily Rate (2020-12)']].mean(axis=1)

rate['revenue_mean'] = rate[['Revenue (2020-11)','Revenue (2020-12)','Revenue (2021-01)','Revenue (2021-02)','Revenue (2021-03)','Revenue (2021-04)',          'Revenue (2021-05)','Revenue (2021-06)','Revenue (2021-07)','Revenue (2021-08)','Revenue (2021-09)','Revenue (2021-10)']].mean(axis=1)

rate['occupancy_mean'] = rate[['Occupancy Rate (2020-11)','Occupancy Rate (2020-12)','Occupancy Rate (2021-01)','Occupancy Rate (2021-02)','Occupancy Rate (2021-03)','Occupancy Rate (2021-04)','Occupancy Rate (2021-05)','Occupancy Rate (2021-06)','Occupancy Rate (2021-07)','Occupancy Rate (2021-08)','Occupancy Rate (2021-09)','Occupancy Rate (2021-10)']].mean(axis=1)


# In[ ]:


#If we have time we can merge rate and listing together for a prediction based on
#https://www.kaggle.com/code/brittabettendorf/what-factors-influence-demand-eda-occupancy
rate.fillna({'Bedrooms': '0'}, inplace = True)


# In[ ]:


#Dropping not needed columns for predicting the rate
#rate.drop(['Listing URL', 'Property Type', 'Has pool'], axis=1, inplace = True)


# In[ ]:


X = rate.drop(['occupancy_mean'], axis=1).values
y = rate['occupancy_mean'].values


# In[ ]:


print(X)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=0)


# In[ ]:


lr = LinearRegression()
lr.fit(X_train, y_train)


# In[ ]:


y_pred = lr.predict(X_test)
print(y_pred)


# In[ ]:


#I am trying different possibilities using different variables, at the moment this one is only the occupancy_mean
r2_score(y_test, y_pred)


# In[ ]:


plt.figure(figsize=(8,4))
plt.scatter(y_test, y_pred)
plt.xlabel('Occupancy at the moment')
plt.ylabel('Occupancy predicted')
plt.title('Occupancy rate')


# In[ ]:


#Predicted values
occu_rate = pd.DataFrame({'Current occupancy': y_test, 'Predicted occupancy': y_pred, 'Profit/Loss': y_test - y_pred})
occu_rate


# In[ ]:


#https://jakevdp.github.io/PythonDataScienceHandbook/03.07-merge-and-join.html
#http://insideairbnb.com/ireland/
#https://public.opendatasoft.com/explore/dataset/airbnb-listings/export/?disjunctive.host_verifications&disjunctive.amenities&disjunctive.features&q=ireland
#https://app.airbtics.com/airbnb-data/ireland/0/dublin
#https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html
#https://www.codegrepper.com/code-examples/python/convert+object+to+float64+pandas
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
#https://stackoverflow.com/questions/48190843/issue-defining-kneighborsclassifier-in-jupyter-notebooks
#https://realpython.com/linear-regression-in-python/
#https://huggingface.co/docs/transformers/v4.18.0/en/installation
#https://huggingface.co/docs/transformers/tokenizer_summary
#https://realpython.com/pandas-merge-join-and-concat/
#https://www.shanelynn.ie/merge-join-dataframes-python-pandas-index-1/
#https://stackoverflow.com/questions/49188960/how-to-show-all-columns-names-on-a-large-pandas-dataframe
#https://www.stackvidhya.com/add-row-to-dataframe/#:~:text=You%20can%20add%20rows%20to,append()
#https://jakevdp.github.io/PythonDataScienceHandbook/03.07-merge-and-join.html#:~:text=2014-,The%20pd.,information%20from%20the%20two%20inputs.


# In[ ]:





# In[ ]:




