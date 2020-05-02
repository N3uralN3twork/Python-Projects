#Practice: http://dybfin.wustl.edu/teaching/optfut/practice/practice4ans.pdf


def square(x):
    return x**2

square(1.1)
#1. Creat
import pandas as pd
names = pd.DataFrame({"First": ["Matt", "Jacob"],
                      "Last": ["Quinn", "Smith"]})


def FullName(df):
    '''To return the full name given a first and last name'''
    import pandas as pd
    return df.First + " " + df.Last


names['Full_Name'] = names.apply(FullName, axis = 1)






# Create a dataframe from a list of dictionaries
#1. Create a dataset / Import one
rectangles = [
    { 'height': 40, 'width': 10 },
    { 'height': 20, 'width': 9 },
    { 'height': 3.4, 'width': 4 }
             ]
#Conver to a pandas DataFrame
rectangles = pd.DataFrame(rectangles)

#2. Define the function you would like to use
def area(row):
    return row['height'] * row.['width']

#3. Apply function to the dataframe
#Axis = 1 applies the function to the rows
rectangles['area'] = rectangles.apply(area, axis = 1)


def area(df):
    return df['area'] = df['height'] * df['width']






"Calculating a Z-score:"
def Z_score(x, mean, std, n=1):
    import numpy as np
    num = x - mean
    denom = std/np.sqrt(n)
    Z = num / denom
    return Z

Z_score(x = 20, mean = 10,
        std = 5,n = 100)



def Black_Scholes(X, S0, div, tte, rf, volatility):
    """
    A model to price European Calls and Puts, I think.
    :param X: the strike price
    :param S0: the current price
    :param div: dividend yield
    :param tte: the time to expiration
    :param rf: the risk-free rate
    :param volatility: the standard deviation
    :return:
    """
    from scipy.stats import norm
    import math
    num = (math.log(S0/X)+(rf-div+((volatility**2)/2))*tte)
    denom = (volatility * math.sqrt(tte))
    d1 = (num / denom)
    d2 = d1 - (volatility * math.sqrt(tte))
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    #C = (S0*Nd1) - (X*(math.e**(-rf(tte/365)))*Nd2)
    C = (S0*Nd1) - (X*(math.exp((-rf*tte))*Nd2))
    B = (X*math.exp(-rf*tte))
    P = (B + C - S0)
    print(f"European Call Price is ${C}")
    print(f"European Put Price is ${P}")

Black_Scholes(X = 50, S0 = 50, div = 0, tte = 0.5, rf = 0.03, volatility = 0.50)
Black_Scholes(X = 45, S0 = 40, div = 0, tte = 1/3, rf = 0.03, volatility = 0.40)
Black_Scholes(X = 120, S0 = 100, div = 0, tte = 0.5, rf = 0.01, volatility = 0.3)
Black_Scholes(X = 50, S0 = 50, div = 0, tte = 4, rf = 0.06, volatility = 0.55)



#To choose which order a class of people goes in.
def RandomShuffle(NumPeople):
    import random
    numbers = random.sample(range(NumPeople), NumPeople)
    return numbers

RandomShuffle(10)