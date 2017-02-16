CONTENTS OF THIS FILE
---------------------
   
 * Introduction
 * Requirements
 * Recommended modules
 * Installation
 * Configuration
 * Troubleshooting
 * FAQ
 * Maintainers
 * How to use it

INTRODUCTION
------------
The MLND Capstone Project 'Investment and Trading' consist in a prediction tool using 'Support Vector Regressor' to analyze historical stock market data. Based on stock symbol inputs, the program provides recommendations for the next day prices (buy or sell) as well as portfolio weight allocation. 

REQUIREMENTS
------------
This program requires the following modules: 
Python 2.7
Matplotlib
Numpy
Sklearn
Pandas
Scipy

INSTALLATION
------------
Install as you would normally installed the tools above. I personally prefer using Anaconda Distribution for the 
installation of the modules mentioned above. 
Platform: Linux (tested), Winddows (not tested)


CONFIGURATION
-------------
Create a /data/ directory at the same level as the files predictor.py and predictor_utilities.py

 ../project/data
 ../project/predictor.py
 ../project/predictor_utilities.py

historical data is saved in /data/ folder for future queries

MAINTAINERS
-----------

Current Maintainers:
 *Ricardo G. Cortez ricortez21@gmail.com



----------------------
HOW TO USE IT
----------------------

>>python ./predictor.py

Select 1 for portfolio analysis by file input: 
Select 2 to input stock symbols: 

#Select 1 for input file. Program takes .txt of .pickle files
#pickle file must contain a python list containing stock symbols
	e.g.
	Select 1 for portfolio analysis by file input: 						
	Select 2 to input stock symbols: 								
	1	
	Input path to portfolio path location:
	/pathtofile/portfolio.txt #where portfolio.txt contains comma separated ticker symbols


				portfolio.txt				
------------------------------------------------------------------------------------------------
WYN, XEC, AAPL, GOOG, ZTS, GM									|
												|
												|
												|
												|
											     eof|
------------------------------------------------------------------------------------------------

#Select 2 for command line ticker symbols input
	e.g.
	Select 1 for portfolio analysis by file input: 
	Select 2 to input stock symbols: 
	2
	Input ticker symbols(s) (comma separated):WYN, XEC, AAPL, GOOG, ZTS, GM


