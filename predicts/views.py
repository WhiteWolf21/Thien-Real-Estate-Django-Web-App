from django.shortcuts import get_object_or_404, render
from django.core.paginator import EmptyPage, PageNotAnInteger, Paginator
from .choices import address_district, address_street, realestate_type, transaction_type, position_street, legal

from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import xgboost as xgb
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors


def getNeighbour(data, sample): # data must be full, but sample must not have output column
  neighbours = data.drop(['post_date', 'price_sell'], axis=1)
  target_sample = sample.drop('post_date')
  neighbours = neighbours[data['address_district'] == target_sample['address_district']]
  neighbours = NearestNeighbors(n_neighbors=min(15,len(neighbours)), algorithm='ball_tree').fit(neighbours)

  return neighbours.kneighbors(target_sample.values.reshape(1,-1), return_distance=False)[0] # return a list of indexes of nearest neighbours

def predictTrend(model, data, sample): #data must be full, but sample must not have output column
  neighbours = data.iloc[getNeighbour(data, sample)] # get most simmilar neighbours
  recent_post_date_mean = neighbours[['post_date','price_sell']].groupby(['post_date'], as_index=False).mean() # group all sample with same post date with mean value
  
  pred_price = model.predict(sample.values.reshape(1,-1))
  past_1_price = recent_post_date_mean['price_sell'].iloc[-1] # get the mean with respective to the recent parameter
  past_2_price = recent_post_date_mean['price_sell'].iloc[-2] # get the mean with respective to the recent parameter

  if past_1_price > past_2_price:
    if pred_price >= past_1_price:
      return [pred_price, list(recent_post_date_mean['price_sell']), list(recent_post_date_mean['post_date']), 1 , round(float(pred_price - past_2_price) * 100 / past_2_price,2) ] # one means increase
    elif pred_price < past_1_price  and pred_price >= past_2_price:
      return [pred_price, list(recent_post_date_mean['price_sell']), list(recent_post_date_mean['post_date']), 0, 0] # zero means unchanged
    else:
      return [pred_price, list(recent_post_date_mean['price_sell']), list(recent_post_date_mean['post_date']), -1 , round((past_2_price - pred_price) * 100 / past_2_price,2)] # minus one means decrease
  elif past_1_price == past_2_price:
    if pred_price > past_1_price:
      return [pred_price, list(recent_post_date_mean['price_sell']), list(recent_post_date_mean['post_date']), 1 , round(float(pred_price - past_2_price) * 100 / past_2_price,2)]
    elif pred_price == past_1_price:
      return [pred_price, list(recent_post_date_mean['price_sell']), list(recent_post_date_mean['post_date']), 0, 0]
    else:
      return [pred_price, list(recent_post_date_mean['price_sell']), list(recent_post_date_mean['post_date']), -1 , round(float(past_2_price - pred_price) * 100 / past_2_price,2)]
  else:
    if pred_price > past_1_price and pred_price <= past_2_price:
      return [pred_price, list(recent_post_date_mean['price_sell']), list(recent_post_date_mean['post_date']), 0, 0]
    elif pred_price <= past_1_price:
      return [pred_price, list(recent_post_date_mean['price_sell']), list(recent_post_date_mean['post_date']), -1 , round(float(past_2_price - pred_price) * 100 / past_2_price,2)]
    else:
      return [pred_price, list(recent_post_date_mean['price_sell']), list(recent_post_date_mean['post_date']), 1 , round(float(pred_price - past_2_price) * 100 / past_2_price,2)]

# from .models import Listing

def predict_result(request):
  # queryset_list = Listing.objects.order_by('-list_date')

  # # Keywords
  # if 'keywords' in request.GET:
  #   keywords = request.GET['keywords']
  #   if keywords:
  #     queryset_list = queryset_list.filter(description__icontains=keywords)

  df = pd.read_csv(r'./data.csv', sep="|")
  df = df.drop("address_street",axis=1)
  df = df.dropna()
  df = df[df['price_sell'] != 0]
  df = df[df['price_sell'] < 1e12]
  df = df[df['floor'] <= 100]
  df = df[df['area'] != 0]

  X = df.drop('price_sell', axis=1)
  y = df['price_sell'].values.ravel()

  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)

  model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=True)
  linear_regr = model.fit(X_train, y_train)
  # linear_regr = model

  now = datetime.now()
  post_date = datetime.timestamp(now)

  test_sample = X_test.iloc[203][:]

  dfData = {
    'address_district': [request.GET['address_district']],
    'realestate_type': [request.GET['realestate_type']],
    'transaction_type': [request.GET['transaction_type']],
    'area': [request.GET['area']],
    'floor': [request.GET['floor']],
    'legal': [request.GET['legal']],
    'position_street': [request.GET['position_street']],
    'post_date': [post_date]
  }

  real_sample = pd.DataFrame(dfData, dtype=float)
  real_sample = real_sample.iloc[0]
  test_sample['post_date'] = post_date

  # print(predictTrend(linear_regr, df, test_sample))

  trend = predictTrend(linear_regr, df, real_sample)

  context = {
    'address_district': address_district,
    'address_street': address_street,
    'realestate_type': realestate_type,
    'transaction_type': transaction_type,
    'position_street': position_street,
    'legal': legal,
    'values': request.GET,
    'price': '{:20,.0f}'.format(int(trend[0])),
    'data': trend[1],
    'label': trend[2],
    'status': trend[3],
    'status_value': trend[4]
  }

  return render(request, 'predicts/predict_result.html', context)