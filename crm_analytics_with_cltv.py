########## CRM Analytics with Cltv ##########

# Variables
# InvoiceNo: Invoice number. The unique number of each transaction, namely the invoice. Delete operation if it starts with C.
# StockCode: Product code. Unique number for each product.
# Description: Product name
# Quantity: Number of products. It expresses how many of the products on the invoices have been sold.
# InvoiceDate: Invoice date and time.
# UnitPrice: Product price
# CustomerID: Unique customer number
# Country: Country name. Country where the customer lives.

################ Data Understanding ################

##### Importing Libraries

import datetime as dt
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from PIL import Image
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


##### Importing Data

# Read the 2010-2011 sheet in the Online Retail II excel.
df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()  # Making a copy of the created dataframe


##### Descriptive Statistics

df.shape  # Dimension of dataframe

df.dtypes  # Data type of each variable

df.info  # Print a concise summary of a DataFrame

df.head()  # First 5 observations of dataframe

df.tail()  # Last 5 observations of dataframe


##### Data Preparation

df.isnull().sum()  # Get number of Null values in a dataframe


# Remove missing observations from the data set
df.dropna(inplace=True)

################ Customer Life Time Value Calculate ################

df = df[~df["Invoice"].str.contains("C", na=False)]  # Delete operation if it starts with C in "Invoice".

df = df[(df['Quantity'] > 0)]

df.dropna(inplace=True)
df["TotalPrice"] = df["Quantity"] * df["Price"]

#  Creation of metrics to be used for the cltv calculate
cltv_c = df.groupby('Customer ID').agg({'Invoice': lambda x: x.nunique(),
                                        'Quantity': lambda x: x.sum(),
                                        'TotalPrice': lambda x: x.sum()})

cltv_c.columns = ['total_transaction', 'total_unit', 'total_price']  # changing column names

cltv_c.head()

############# CLTV_C = (Customer Value/ Churn Rate) * (Profit Margin) #############

# average_order_value = total_price / total_transaction
cltv_c['avg_order_value'] = cltv_c['total_price'] / cltv_c['total_transaction']


# Purchase Frequency = total_transaction / total_number_of_customers
cltv_c["purchase_frequency"] = cltv_c['total_transaction'] / cltv_c.shape[0]


# Repeat Rate & Churn Rate (Number of Customers Who Shopped More Than Once) / (Total Number of Customer)
repeat_rate = cltv_c[cltv_c.total_transaction > 1].shape[0] / cltv_c.shape[0]
churn_rate = 1 - repeat_rate


# Profit Margin (profit_margin =  total_price * 0.10)
cltv_c['profit_margin'] = cltv_c['total_price'] * 0.10


# Customer Value (customer_value = average_order_value * purchase_frequency)
# Customer Value
cltv_c['customer_value'] = (cltv_c['avg_order_value'] * cltv_c["purchase_frequency"]) / churn_rate


# Customer Lifetime Value (CLTV = (customer_value / churn_rate) x profit_margin)
cltv_c['cltv'] = cltv_c['customer_value'] * cltv_c['profit_margin']
cltv_c.head()


# Scaled cltv values
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv_c[["cltv"]])
cltv_c["scaled_cltv"] = scaler.transform(cltv_c[["cltv"]])
cltv_c.sort_values(by="scaled_cltv", ascending=False).head()


# Segmentation of customers into 4 groups by creating segments
cltv_c["segment"] = pd.qcut(cltv_c["scaled_cltv"], 4, labels=["D", "C", "B", "A"])
cltv_c.head()

# Observation of most valuable customers according to scaled_clv
cltv_c[["total_transaction", "total_unit", "total_price", "cltv", "scaled_cltv"]].sort_values(by="scaled_cltv",
                                                                                              ascending=False).head()

# Description of segments:
cltv_c.groupby("segment")[["total_transaction", "total_unit", "total_price", "cltv", "scaled_cltv"]].agg(
    {"count", "mean", "sum"})





################ Customer Life Time Value Prediction ################

# Functions required to delete outliers
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


df = df_.copy()
df.head()

# Remove missing observations from the data set
df.dropna(inplace=True)

df = df[~df["Invoice"].str.contains("C", na=False)]  # Delete operation if it starts with C in "Invoice".
df = df[df["Quantity"] > 0]

# Deleting outliers in "Quantity" and "Price" variables
replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")


df["TotalPrice"] = df["Quantity"] * df["Price"]
today_date = dt.datetime(2011, 12, 11)


# Cltv Prediction Veri Yapısının Hazırlanması

# recency: The time passed since the customer's first and last purchase.
# T: Customer's age
# frequency: Total number of repeat purchases(frequency>1)
# monetary_value: Average price per purchase

#  Creation of metrics to be used for the cltv predict
cltv_df = df.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days],
                                         'Invoice': lambda num: num.nunique(),
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

cltv_df.head()

# changing column names
cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

cltv_df.head()

# Expressing "monetary" value as average earnings per purchase
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

# Choosing greater than zero for "monetary"
cltv_df = cltv_df[cltv_df["monetary"] > 0]
cltv_df.head()

# Weekly value of "recency" and "T" for BGNBD
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7

cltv_df.head()

# Choosing greater than one for "frequency"
cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

cltv_df.head()


# BG-NBD Model

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])


# Examples
# Who are the 10 customers we expect the most to purchase in 1 month?

bgf.predict(4,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)

cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])
cltv_df.head()

# Who are the 10 customers we expect to purchase the most in 12 months?

bgf.predict(4*12,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)

cltv_df["expected_purc_12_month"] = bgf.predict(4*12,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])
cltv_df.head()


# GAMMA-GAMMA Model


ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])


# Calculation of CLTV with BG-NBD and GG model.


cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  # monthly
                                   freq="W",
                                   discount_rate=0.01)

cltv.head()

cltv.shape
cltv = cltv.reset_index()
cltv.sort_values(by="clv", ascending=False).head(50)

cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_final.sort_values(by="clv", ascending=False).head(10)

# Scaled cltv
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv_final[["clv"]])
cltv_final["scaled_clv"] = scaler.transform(cltv_final[["clv"]])

cltv_final.head()


# Let's sort
cltv_final.sort_values(by="scaled_clv", ascending=False).head()


# Segmentation of customers into 4 groups by creating segments
cltv_final["segment"] = pd.qcut(cltv_final["scaled_clv"], 4, labels=["D", "C", "B", "A"])
cltv_final.head()

cltv_final.sort_values(by="scaled_clv", ascending=False).head(10)


# Description of segments:
cltv_final.groupby("segment").agg(
    {"count", "mean", "sum"})

cltv_final.groupby("segment").agg(
    {"recency": ["count", "mean", "sum"],
     "T": ["count", "mean", "sum"],
     "frequency": ["count", "mean", "sum"],
     "monetary": ["count", "mean", "sum"],
     "scaled_clv": ["count", "mean", "sum"]})

