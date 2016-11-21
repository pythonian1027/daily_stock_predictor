
# coding: utf-8

# # Machine Learning Engineer Nanodegree
# ## Model Evaluation & Validation
# ## Project 1: Predicting Boston Housing Prices
# 
# Welcome to the first project of the Machine Learning Engineer Nanodegree! In this notebook, some template code has already been provided for you, and you will need to implement additional functionality to successfully complete this project. You will not need to modify the included code beyond what is requested. Sections that begin with **'Implementation'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section and the specifics of the implementation are marked in the code block with a 'TODO' statement. Please be sure to read the instructions carefully!
# 
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.  
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ## Getting Started
# In this project, you will evaluate the performance and predictive power of a model that has been trained and tested on data collected from homes in suburbs of Boston, Massachusetts. A model trained on this data that is seen as a *good fit* could then be used to make certain predictions about a home — in particular, its monetary value. This model would prove to be invaluable for someone like a real estate agent who could make use of such information on a daily basis.
# 
# The dataset for this project originates from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Housing). The Boston housing data was collected in 1978 and each of the 506 entries represent aggregated data about 14 features for homes from various suburbs in Boston, Massachusetts. For the purposes of this project, the following preprocessing steps have been made to the dataset:
# - 16 data points have an `'MDEV'` value of 50.0. These data points likely contain **missing or censored values** and have been removed.
# - 1 data point has an `'RM'` value of 8.78. This data point can be considered an **outlier** and has been removed.
# - The features `'RM'`, `'LSTAT'`, `'PTRATIO'`, and `'MDEV'` are essential. The remaining **non-relevant features** have been excluded.
# - The feature `'MDEV'` has been **multiplicatively scaled** to account for 35 years of market inflation.
# 
# Run the code cell below to load the Boston housing dataset, along with a few of the necessary Python libraries required for this project. You will know the dataset loaded successfully if the size of the dataset is reported.

# In[3]:

# Import libraries necessary for this project
import numpy as np
import pandas as pd
import visuals as vs # Supplementary code
from sklearn.cross_validation import ShuffleSplit

# Pretty display for notebooks
get_ipython().magic(u'matplotlib inline')

# Load the Boston housing dataset
data = pd.read_csv('housing.csv')
prices = data['MDEV']
features = data.drop('MDEV', axis = 1)
    
# Success
print "Boston housing dataset has {} data points with {} variables each.".format(*data.shape)


# ## Data Exploration
# In this first section of this project, you will make a cursory investigation about the Boston housing data and provide your observations. Familiarizing yourself with the data through an explorative process is a fundamental practice to help you better understand and justify your results.
# 
# Since the main goal of this project is to construct a working model which has the capability of predicting the value of houses, we will need to separate the dataset into **features** and the **target variable**. The **features**, `'RM'`, `'LSTAT'`, and `'PTRATIO'`, give us quantitative information about each data point. The **target variable**, `'MDEV'`, will be the variable we seek to predict. These are stored in `features` and `prices`, respectively.

# ### Implementation: Calculate Statistics
# For your very first coding implementation, you will calculate descriptive statistics about the Boston housing prices. Since `numpy` has already been imported for you, use this library to perform the necessary calculations. These statistics will be extremely important later on to analyze various prediction results from the constructed model.
# 
# In the code cell below, you will need to implement the following:
# - Calculate the minimum, maximum, mean, median, and standard deviation of `'MDEV'`, which is stored in `prices`.
#   - Store each calculation in their respective variable.

# In[4]:

# TODO: Minimum price of the data
minimum_price = min(prices)

# TODO: Maximum price of the data
maximum_price = max(prices)

# TODO: Mean price of the data
mean_price = np.mean(prices)

# TODO: Median price of the data
median_price = np.median(prices)

# TODO: Standard deviation of prices of the data
std_price = np.std(prices)

# Show the calculated statistics
print "Statistics for Boston housing dataset:\n"
print "Minimum price: ${:,.2f}".format(minimum_price)
print "Maximum price: ${:,.2f}".format(maximum_price)
print "Mean price: ${:,.2f}".format(mean_price)
print "Median price ${:,.2f}".format(median_price)
print "Standard deviation of prices: ${:,.2f}".format(std_price)


# ### Question 1 - Feature Observation
# As a reminder, we are using three features from the Boston housing dataset: `'RM'`, `'LSTAT'`, and `'PTRATIO'`. For each data point (neighborhood):
# - `'RM'` is the average number of rooms among homes in the neighborhood.
# - `'LSTAT'` is the percentage of all Boston homeowners who have a greater net worth than homeowners in the neighborhood.
# - `'PTRATIO'` is the ratio of students to teachers in primary and secondary schools in the neighborhood.
# 
# _Using your intuition, for each of the three features above, do you think that an increase in the value of that feature would lead to an **increase** in the value of `'MDEV'` or a **decrease** in the value of `'MDEV'`? Justify your answer for each._  
# **Hint:** Would you expect a home that has an `'RM'` value of 6 be worth more or less than a home that has an `'RM'` value of 7?

# **Answer: 
# -An increase in the average number of rooms ('RM') would, very likely, lead to an increase in the value of 'MDEV' because it could be an indicator of a bigger house (more square footage) and therefore higher price. 
# -An increase in 'LSTAT' would lead to a decrease in the value of 'MDEV' because it is an indicator of the economic status of the neighborhood in comparisson with all Boston homeowners. Therefore, an increase in 'LSTAT' indicates that there is a larger population of Boston homeowners with a greater net worth that homeowners in the neighborhood. A decrease in 'LSTAT' would indicate the the net worth of homeowners in the neighborhood is greater than the one of all Boston homeowners which would indicate a more valuable area. 
# -An increase in 'PTRATIO' would lead to a decrease in 'MDEV' because it indicated that there are more students per teacher. Homeowners tend to look for more personalized schools therefore a lower 'PTRATIO' is preferred. 

# ----
# 
# ## Developing a Model
# In this second section of the project, you will develop the tools and techniques necessary for a model to make a prediction. Being able to make accurate evaluations of each model's performance through the use of these tools and techniques helps to greatly reinforce the confidence in your predictions.

# ### Implementation: Define a Performance Metric
# It is difficult to measure the quality of a given model without quantifying its performance over training and testing. This is typically done using some type of performance metric, whether it is through calculating some type of error, the goodness of fit, or some other useful measurement. For this project, you will be calculating the [*coefficient of determination*](http://stattrek.com/statistics/dictionary.aspx?definition=coefficient_of_determination), R<sup>2</sup>, to quantify your model's performance. The coefficient of determination for a model is a useful statistic in regression analysis, as it often describes how "good" that model is at making predictions. 
# 
# The values for R<sup>2</sup> range from 0 to 1, which captures the percentage of squared correlation between the predicted and actual values of the **target variable**. A model with an R<sup>2</sup> of 0 always fails to predict the target variable, whereas a model with an R<sup>2</sup> of 1 perfectly predicts the target variable. Any value between 0 and 1 indicates what percentage of the target variable, using this model, can be explained by the **features**. *A model can be given a negative R<sup>2</sup> as well, which indicates that the model is no better than one that naively predicts the mean of the target variable.*
# 
# For the `performance_metric` function in the code cell below, you will need to implement the following:
# - Use `r2_score` from `sklearn.metrics` to perform a performance calculation between `y_true` and `y_predict`.
# - Assign the performance score to the `score` variable.

# In[5]:

# TODO: Import 'r2_score'

def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    from sklearn.metrics import r2_score
    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    score = r2_score(y_true, y_predict)
    
    # Return the score
    return score


# ### Question 2 - Goodness of Fit
# Assume that a dataset contains five data points and a model made the following predictions for the target variable:
# 
# | True Value | Prediction |
# | :-------------: | :--------: |
# | 3.0 | 2.5 |
# | -0.5 | 0.0 |
# | 2.0 | 2.1 |
# | 7.0 | 7.8 |
# | 4.2 | 5.3 |
# *Would you consider this model to have successfully captured the variation of the target variable? Why or why not?* 
# 
# Run the code cell below to use the `performance_metric` function and calculate this model's coefficient of determination.

# In[6]:

# Calculate the performance of this model
score = performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
print "Model has a coefficient of determination, R^2, of {:.3f}.".format(score)


# **Answer:0.923. The model successfully captures the variation of the target variable. By simple inspection, one can observe a low error generated between True and Predicted values, intuitively one can say that the model is performing well. The high r2_score (range from 0 to 1, with 1 being perfect accuracy) confirms our intuition that the model if succefully capturing the variation of the target variable.

# ### Implementation: Shuffle and Split Data
# Your next implementation requires that you take the Boston housing dataset and split the data into training and testing subsets. Typically, the data is also shuffled into a random order when creating the training and testing subsets to remove any bias in the ordering of the dataset.
# 
# For the code cell below, you will need to implement the following:
# - Use `train_test_split` from `sklearn.cross_validation` to shuffle and split the `features` and `prices` data into training and testing sets.
#   - Split the data into 80% training and 20% testing.
#   - Set the `random_state` for `train_test_split` to a value of your choice. This ensures results are consistent.
# - Assign the train and testing splits to `X_train`, `X_test`, `y_train`, and `y_test`.

# In[7]:

# TODO: Import 'train_test_split'
from sklearn.cross_validation import train_test_split
# TODO: Shuffle and split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size = 0.2, random_state = 42)

# Success
print "Training and testing split was successful."
#print X_train, y_test
# ### Question 3 - Training and Testing
# *What is the benefit to splitting a dataset into some ratio of training and testing subsets for a learning algorithm?*  
# **Hint:** What could go wrong with not having a way to test your model?

# **Answer:Splitting the dataset into subsets eliminates bias. The model is trained with the training set and tested with the test set, which is different from the training set. If we use the same data points to test and train then results will tend to indicate a high accuracy but when real with different data results the accuracy will be much lower because it is a more realistic testing environment. 

# ----
# 
# ## Analyzing Model Performance
# In this third section of the project, you'll take a look at several models' learning and testing performances on various subsets of training data. Additionally, you'll investigate one particular algorithm with an increasing `'max_depth'` parameter on the full training set to observe how model complexity affects performance. Graphing your model's performance based on varying criteria can be beneficial in the analysis process, such as visualizing behavior that may not have been apparent from the results alone.

# ### Learning Curves
# The following code cell produces four graphs for a decision tree model with different maximum depths. Each graph visualizes the learning curves of the model for both training and testing as the size of the training set is increased. Note that the shaded region of a learning curve denotes the uncertainty of that curve (measured as the standard deviation). The model is scored on both the training and testing sets using R<sup>2</sup>, the coefficient of determination.  
# 
# Run the code cell below and use these graphs to answer the following question.

# In[6]:

# Produce learning curves for varying training set sizes and maximum depths
vs.ModelLearning(features, prices)


# ### Question 4 - Learning the Data
# *Choose one of the graphs above and state the maximum depth for the model. What happens to the score of the training curve as more training points are added? What about the testing curve? Would having more training points benefit the model?*  
# **Hint:** Are the learning curves converging to particular scores?

# **Answer: max_depth = 3. The score of the training curve slightly decreases as the number of training points increase (but it seems to set plateau). In the training curve, the model learns from the environment but does not fully capture the underlying complexity of the data as more training points are used. The test curve increases as the number of training points are increased indicating a high variance behavior in the initial points but converging as the number of training points increases.
# Would having more training points benefit the model? No, both testing and training curves have set plateau and have converged, adding more training data points won't benefit the model. 

# ### Complexity Curves
# The following code cell produces a graph for a decision tree model that has been trained and validated on the training data using different maximum depths. The graph produces two complexity curves — one for training and one for validation. Similar to the **learning curves**, the shaded regions of both the complexity curves denote the uncertainty in those curves, and the model is scored on both the training and validation sets using the `performance_metric` function.  
# 
# Run the code cell below and use this graph to answer the following two questions.

# In[12]:

vs.ModelComplexity(X_train, y_train)


# ### Question 5 - Bias-Variance Tradeoff
# *When the model is trained with a maximum depth of 1, does the model suffer from high bias or from high variance? How about when the model is trained with a maximum depth of 10? What visual cues in the graph justify your conclusions?*  
# **Hint:** How do you know when a model is suffering from high bias or high variance?

# **Answer: When the maximum depth is 1, the model suffers from high bias (underfit), the model is not complex enough to understand the underlying complexity of the data. In other words, although the performance for the validation and training curves are almost the same at this point, the performance is low, around 0.4, indicating that the model won't benefit from more data; provided that depth equals to one. Adding complexity to the model, in this case increasing the depth, would help the model to understand better the data. At maximum depth equals to 10, the model suffers from high variance, there is a large gap between training score and Validation score indicating that model is not able to generalize its predictions accurately. Based on the "Representative Power of a Model" chapter, as the number of features or dimensions grow, the amount of training points grows exponentially, this notion can be used to explain why it performs better when using training data but decays with testing data. First, note that the curve is for a fix number of training points, adding more complexity (increase the depth) can be explained as if the model now is able to 'capture' more features from the model (for instance, if we have features for different cars: 'Make', 'Model', 'Year', 'Odometer Reading', if a prediction model wants to predict the mpg based on a decision tree with 1 layer -using the feature 'Make'- then it won't be accurate, but if we add a second layer -adding the feature 'Model'- then the model will have more insights on the data, if, subsequently, we add the other layers: correnponding to 'Year' and 'Odometer Readings', then the model will be able to 'capture' the dimensions (features) of the data). Another way to explain it is by looking at the curve, it reaches its maximum performance at 3 (or 4) because that's the total number of features, but as we keep increasing the number of layers, the model becomes more complex (overfit) and starts making false assumptions. (I'm not necessarily stating that the model needs as many layers as features but that there is a correlation). At depth equals 10, the model performs poorly when generalizing, the model has become more complex than it needs to be. At depth equals 3, as mentioned in question 4, the model won't benefit from more data training points because at this point, both have converged and reached plateau.

# ### Question 6 - Best-Guess Optimal Model
# *Which maximum depth do you think results in a model that best generalizes to unseen data? What intuition lead you to this answer?*

# **Answer: Depth of 3. Because, the training a Validations performance are very similar, so it could be considered that it is (at this point) a generalized model that can perform very similarly under training or testing data. After this point, the gap between testing and validation increases and Validations performance start to decay. 

# -----
# 
# ## Evaluating Model Performance
# In this final section of the project, you will construct a model and make a prediction on the client's feature set using an optimized model from `fit_model`.

# ### Question 7 - Grid Search
# *What is the grid search technique and how it can be applied to optimize a learning algorithm?*

# **Answer: Thre grid search technique is the way to iterate through an array of input parameters and chose the optimal ones. In a learning algorithm, it is cumversome to manually apply different parameters and analyze the performance of each individually before finding an optimal set of them for our model. The grid search technique automates this approach making it easier for the developer since it performs the calculations under the hood and only outputs the set of parameters that maximizes the performance. 

# ### Question 8 - Cross-Validation
# *What is the k-fold cross-validation training technique? What benefit does this technique provide for grid search when optimizing a model?*  
# **Hint:** Much like the reasoning behind having a testing set, what could go wrong with using grid search without a cross-validated set?

# **Answer: The k-fold cross-validation training technique is a way to validate the performance using all the data available. The testing set is chosen multiple times from the data available, so that at the end all the data has been used for testing, and the overall performance is an average of each of the individual performances. This approach benefits the grid search in the sense that while grid search calculates the performance under multiple input parameters, by using k-fold cross-validation, the model is been tested under a larger set of case, so that the overall result is a more sounded, generalized performance. 

# ### Implementation: Fitting a Model
# Your final implementation requires that you bring everything together and train a model using the **decision tree algorithm**. To ensure that you are producing an optimized model, you will train the model using the grid search technique to optimize the `'max_depth'` parameter for the decision tree. The `'max_depth'` parameter can be thought of as how many questions the decision tree algorithm is allowed to ask about the data before making a prediction. Decision trees are part of a class of algorithms called *supervised learning algorithms*.
# 
# For the `fit_model` function in the code cell below, you will need to implement the following:
# - Use [`DecisionTreeRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) from `sklearn.tree` to create a decision tree regressor object.
#   - Assign this object to the `'regressor'` variable.
# - Create a dictionary for `'max_depth'` with the values from 1 to 10, and assign this to the `'params'` variable.
# - Use [`make_scorer`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html) from `sklearn.metrics` to create a scoring function object.
#   - Pass the `performance_metric` function as a parameter to the object.
#   - Assign this scoring function to the `'scoring_fnc'` variable.
# - Use [`GridSearchCV`](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html) from `sklearn.grid_search` to create a grid search object.
#   - Pass the variables `'regressor'`, `'params'`, `'scoring_fnc'`, and `'cv_sets'` as parameters to the object. 
#   - Assign the `GridSearchCV` object to the `'grid'` variable.

# In[8]:

# TODO: Import 'make_scorer', 'DecisionTreeRegressor', and 'GridSearchCV'
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)
#    print cv_sets.n_test
    # TODO: Create a decision tree regressor object
    regressor = DecisionTreeRegressor()

    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth':[1,2,3,4,5,6,7,8,9,10] }

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 
        
    scoring_fnc = make_scorer(performance_metric)

    # TODO: Create the grid search object
    grid = GridSearchCV(regressor, params, scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_


# ### Making Predictions
# Once a model has been trained on a given set of data, it can now be used to make predictions on new sets of input data. In the case of a *decision tree regressor*, the model has learned *what the best questions to ask about the input data are*, and can respond with a prediction for the **target variable**. You can use these predictions to gain information about data where the value of the target variable is unknown — such as data the model was not trained on.

# ### Question 9 - Optimal Model
# _What maximum depth does the optimal model have? How does this result compare to your guess in **Question 6**?_  
# 
# Run the code block below to fit the decision tree regressor to the training data and produce an optimal model.

# In[9]:

# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

# Produce the value for 'max_depth'
print "Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth'])


# Parameter 'max_depth' is 4 for the optimal model. 
# How does this result compare to your guess in Question 6?
# They both have very similar validation scores, around 0.74. The model's gridsearch has iterate through all the possible depths and found the optimal, therefore, depth equals to 4 might be slightly better than depth equals to 3 based on R2 metric. 

# ### Question 10 - Predicting Selling Prices
# Imagine that you were a real estate agent in the Boston area looking to use this model to help price homes owned by your clients that they wish to sell. You have collected the following information from three of your clients:
# 
# | Feature | Client 1 | Client 2 | Client 3 |
# | :---: | :---: | :---: | :---: |
# | Total number of rooms in home | 5 rooms | 4 rooms | 8 rooms |
# | Household net worth (income) | Top 34th percent | Bottom 45th percent | Top 7th percent |
# | Student-teacher ratio of nearby schools | 15-to-1 | 22-to-1 | 12-to-1 |
# *What price would you recommend each client sell his/her home at? Do these prices seem reasonable given the values for the respective features?*  
# **Hint:** Use the statistics you calculated in the **Data Exploration** section to help justify your response.  
# 
# Run the code block below to have your optimized model make predictions for each client's home.

# In[11]:

# Produce a matrix for client data
client_data = [[5, 34, 15], # Client 1
               [4, 55, 22], # Client 2
               [8, 7, 12]]  # Client 3

# Show predictions
for i, price in enumerate(reg.predict(client_data)):
    print "Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price)


# **Answer: **
# Recommended values based on model:
# 
# Predicted selling price for Client 1's home: $344,400
# This value does not seem reasobable when compared with the parameters of clien2. It posses a lower 'LSTAT' and 'PTAM' than client2, which would correspond to a higher price, but I would expect this value to be closer to client2's value. 
# 
# Predicted selling price for Client 2's home: $237,478.72
# Given high number of PTRATIO and relatively low RM, I would expect a lower value than client1's but the predicted value seems too low. A point to note is that when changing 'PTAM' to a lower value (e.g. 12) the predicted value decreases instead of increase as I would expect.
# 
# Predicted selling price for Client 3's home: $976,150.00
# For client 3, this value seems reasonable based on the Data Exploration, the RM, LSTAT and PTRATIO were : 8.398, 5.91 and 13 respectively; very similar to client3's values. 
# 
# 

# ### Sensitivity
# An optimal model is not necessarily a robust model. Sometimes, a model is either too complex or too simple to sufficiently generalize to new data. Sometimes, a model could use a learning algorithm that is not appropriate for the structure of the data given. Other times, the data itself could be too noisy or contain too few samples to allow a model to adequately capture the target variable — i.e., the model is underfitted. Run the code cell below to run the `fit_model` function ten times with different training and testing sets to see how the prediction for a specific client changes with the data it's trained on.

# In[142]:

vs.PredictTrials(features, prices, fit_model, client_data)


# ### Question 11 - Applicability
# *In a few sentences, discuss whether the constructed model should or should not be used in a real-world setting.*  
# **Hint:** Some questions to answering:
# - *How relevant today is data that was collected from 1978?*
# - *Are the features present in the data sufficient to describe a home?*
# - *Is the model robust enough to make consistent predictions?*
# - *Would data collected in an urban city like Boston be applicable in a rural city?*

# **Answer: **
# The constructed model shouldn't be used in a real-world setting because it was based on information that has change over the years. Even though the MDEV has been scaled to account for 35 yeard of inflation, the other parameters must likely have changed over time and are not reflecting the current real state market. For instance, the ratio students per teacher has most likely changed, also the neighborhoods net worths tend to fluctuate, not rapidly, but it would be a factor to consider in a period of 37 years. 
# The features present in the data are not sufficient to describe a home, the constructed and lot areas are important factors as well as the location (whether is a suburb or in the city). Describing a home could take many more parameters such as the number of bathrooms, size of the garage, whether or is part of a gated community, proximity to public transportation, age of the house, number of previous owners, previous sales prices, etc. 
# The model is not robus to make consistent predictions, as it was pointed in the previous exercise, the model is not capturing the underlying behavior of the data. The variation of prices between trials is too high indicating that the model is not representing the target accurately and it either requires more data (underfit) or a more complex model is required (overfit). 
# Data collected in an urban city like Boston would not be applicable in a rural city, not even if it's data from 1978 (one could be tempted to consider that a city now was a rural city 37 years ago but this is not the case for Boston, an old city). Also, rural areas tend to have larger house lots and more spreadout layouts, I would consider the mechanics behind the prediction of a city like Boston are totally different from the ones in a rural city. 
# 