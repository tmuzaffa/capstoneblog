# Starbucks Promotional Strategy: A Machine Learning Approach
![png](1.png)


There has been an exponential increase in the use of Artificial Intelligence (AI) and Machine Learning (ML) in the marketing  strategy by businesses around the globe. AI and ML  have impacted not just marketing and sales but various other field in an unprecedented way (Siau & Yang, 2017). Use of metadata and social networks for promotions have increased  and have led to increased profits (Sundsoy, Bjelland, Asif, Pentalnd, & Yves-Alexandre). 
In this project, we developed a  machine learning approach to analyze the customer behavior on the Starbucks reward mobile app. The data set consisted of simulated data that mimics the customer behavior. ML approach used Principle Component Analysis (PCA) (Dezyre.com, n.d.), XGBRegressor (Xgboost, n.d.), and GridSearchCV (Scikitlearn, n.d.) for prediction. 
The data for this project was provided by Starbucks Inc. Starbucks data set consisted of three files namely

- portfolio.json
- profile.json
- transaction.json

portfolio.json contained metadata about each offer, profile.json contained demographic data of users, and transcript.json contained transactional data. 
#### Portfolio.json
portfolio.json consisted of below features
-	channels (methods used to send offer to users)
-	difficulty (minimum spending before offer applies)
-	duration (duration validity of offer)
-	id  (offer id)
-	offer_type  (type of offer sent  to users)
-	reward ( reward to be applied during valid offer duration after difficulty level achieved)
#### profile.json:
profile.json consisted of below features
-	age (age of user)
-	became_member_on (user became member on that day)
-	gender (gender of user)
-	id ( user id)
-	income (income of user)
#### transcript.json:
transcript.json consisted of below features
-	event ( kind of event occurred, e.g. offer received, offer viewed, transaction made etc.)
-	person (user id)
-	time (time at which event occurred)
-	value ( monetary value of the transaction)

The objective of the project is to combine transaction, demographic, and offer data to determine which demographic groups respond best to which offer type. This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products.

### Problem Statement:

The goal is to determine which demographics respond best to what type of offer so that they can be sent only that type of offer to maximize profitability.  We also want to make sure, not to send offers to customers who are already loyal to Starbucks brand as they will make purchases with or without offer, and to the customers who might stop purchasing because of offers as they might not want to participate in the company marketing campaign. The task involved in this project are the following:
1)	Pre-process portfolio.json, profile.json, and transcript.json
2)	Predict age, gender, and income distribution of missing data using gradient boosted tree regressor and classifier. 
3)	Merge predicted data for missing values with known data
4)	Do feature engineering to calculate monthly profits made for each offer type
5)	Analyze monthly profits for each offer type and look at trends
The final analysis are expected to shed light on the type of offer and demographics that Starbucks should target for maximum profitability.

### Metrics:
 Metric used to see the best offer is monthly profit and number of transactions in that particular month. 
Metrics used for data pre-processing are below
•	Root Mean Squared Error (RMSE) (Wikipedia, n.d.)
RMSE is a commonly used to optimize the models. Optimization of model involve minimizing RMSE 
RMSE is a commonly used to optimize the models. Optimization of model involve minimizing RMSE 

RMSE= √((∑_(i=1)^n▒(y ̂_t-y_t ) )/n)   ,

#### BUSINESS UNDERSTANDING

The data sets obtained for the analysis of  vancouver air bnb market contain property listing data set, calendar data set, and reviews data set. All these data set gives us insights about the Vancouver air bnb market. We will use these data set in tandem to find corrleations and answer the following questions.  

Pricing correlation:
* How does price correlates with seasons of year?
* How the type of property impacts the listing price in Vancouver??
* Dependance of listing price on the neighbourhoods in Vancouver?



Analysis of Reviews:
- Understanding how reviews impact the occupancy rate
- Get a correlation of the reviews with the neighbourhoods in vancouver
- Can we explore some of the worst reviews for additional insights?


Influence of parameters on availability:
- Cancellation policy
- Room type
- Number of guests
- Guests picture





    

### Data set review

Now we look at all the data sets to find their characteristics and if they have null values, we will is.null() functionality to check the missing values and will use the .describe() to get some sense of features for each column.

Questions to be answered for data review
1. Are there any missing values?
2. What are the features of each column?


Below analysis show that the vancouver hosts respond very efficiently


![png](output_20_0.png)


It can be seen that approximately 70% of the hosts in metro vancouver respond with in an hour. There are various types of properties available in Vancouver in the Air bnb platform. Below is the breakdown


![png](output_24_0.png)

Approximately 35% of the properties listed on Vancouver Air bnb are houses and 26% are apartments

### Question 1 - Price correlations
1. Understand the correlation between price and the season of the year, and detect the peak season in Vancouver
2. To get an understanding of price correlation with the neighbourhoods in vancouver
3. Getting an insight into the relationship between the listed pricing and the property type in metro vancovuer




![png](output_29_0.png)

July-September seems to be the peak season in Vancouver




![png](output_30_0.png)


As we can see that listing price is higher for the number of reviews between 200-300, as the number of reviews increase we can see that price drops, which suggests that more customer tend to review places which are economical, however places with 200-300 tend to have higher listing price. 

Now we look at the neighbourhood and their pricings




    


![png](output_32_1.png)





![png](output_33_0.png)






![png](output_34_0.png)





It can be seen from the charts above that the highest average price listed per night is $250 and cheapest is $120 per night in metro Vancouver.





![png](output_40_0.png)



It can be seen from the heatmap above that Houses in downtown Vancouver area have higher listing price than apartments. 


### Question 02 - Review correlation

- Understanding how reviews impact the occupancy rate
- Get a correlation of the reviews with the neighbourhoods in vancouver
- Can we explore some of the worst reviews for additional insights?

let us look at the data in the review and listing table and also calcualte the occupancy rate from the listing can calendar table


![png](output_48_1.png)



Review rates are higher for properties with high occupancy rate. This could be either bad review or good review



![png](output_57_0.png)






![png](output_59_0.png)


### Question # 03

Now we will look at the influence of parameters such as below on the availability rate 
- Cancellation policy
- Room type
- Number of guests
- Guests picture


First we try to clean the data and deal with the missing values, For bathrooms column, there are two missing values, but a property must have atleast 1 bathroom, so we will use 1 to fill the NA. We will also drop unnecessary columns






![png](output_74_1.png)








![png](output_77_1.png)


##### Results

Botique hotels are the least occupied property in metro vancouver


![png](output_80_1.png)







Above results suggest that, properties with more moderate to flexible cancellation policy tend to have more demand and higher occupancy, whereas properties with more strict cancellation policy have more availability. This suggests that for higher profits, businesses should move towards more relaxed cancellation policy





![png](output_83_1.png)




Listings which can host from families upto a small group are high in demand







![png](output_86_1.png)




Properties that donot require guests profile photo for booking  has low availability and are more popular


## Summary Of Conclusions<a name="results"></a>

Key findings from the analysis are summarized below:

1. It was found that approximately 70% of the hosts in metro vancouver respond with in an hour. 
2. Approximately 35% of the properties listed on Vancouver Air bnb are houses and 26% are apartments.
3. Listing price is higher for the number of reviews between 200-300, as the number of reviews increases, price drops, which suggests that more customer tend to review places which are economical, however places with 200-300 tend to have higher listing price. 

4. Downtown, Kistilano are the most expensive neighbourhoods, whereas Killarnay is the cheapest.
5. We didnt see any storng relationship with the occupancy rate and the review scores.
6. We also found that Kistilano and Downtown east neighbourhoods receivee the most positive reviews. 
7. We also found that there is a storng relationship between property type, room type, number of guests with the occupancy rate.


## Acknowledgements<a name="acknowledgements"></a>

- Credit to the AirBnB dataset published by AirBnB and Kaggle for hosting it, the dataset here: https://www.kaggle.com/airbnb/
- Remove the plot border: https://stackoverflow.com/questions/925024/how-can-i-remove-the-top-and-right-axis-in-matplotlib
- Annotations:  https://robertmitchellv.com/blog-bar-chart-annotations-pandas-mpl.html
- Gradient Color: https://www.pythonprogramming.in/bar-chart-with-different-color-of-bars.html
- Remove the $ symbol: https://stackoverflow.com/questions/22588316/pandas-applying-regex-to-replace-values
- Subplots: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.subplots.html
- SentimentIntensityAnalyzer:https://stackoverflow.com/questions/39462021/nltk-sentiment-vader-polarity-scorestext-not-working
- Dropping multiple columns: https://stackoverflow.com/questions/28538536/deleting-multiple-columns-based-on-column-names-in-pandas                                  -  https://stackoverflow.com/questions/17838752/how-to-delete-multiple-columns-in-one-pass
- Dtype:  https://stackoverflow.com/questions/21271581/selecting-pandas-columns-by-dtype
- Replacing True, False: https://stackoverflow.com/questions/23307301/replacing-column-values-in-a-pandas-dataframe

