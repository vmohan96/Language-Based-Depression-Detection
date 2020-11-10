#!/usr/bin/env python
# coding: utf-8

# # 9/29 - Reddit API
# ------
# **Author:** Sara Soueidan<br>
# **Source:** Sara Soueidan - Project 3, Tim Book - API Code
# 
# We will (1) examine the Reddit API and Pushshift, (2) test Pushshift with a small call and (3) build a function that outputs a DataFrame with posts info from specific subreddit.
# 
# _Disclaimer: Reddit posts may contain content NSFW._

# ## Part 1 - Examine API
# [Reddit API Documentation](https://www.reddit.com/dev/api/)<br>
# [Pushshift API Wrapper Documentation](https://pushshift.io/api-parameters/)

# ## Part 2 - Test API
# [Epoch Extractor](https://www.epochconverter.com/)

# **Imports**

# In[1]:


# Standards
import pandas as pd
import numpy as np
import time

# API
import requests

# Automating
import time
import datetime
import warnings
import sys


# **Test Pull from World of Warcraft Subreddit**

# **Write Function**

# In[16]:


def get_posts(subreddit, n_iter, epoch_right_now): # subreddit name and number of times function should run
    # store base url variable
    base_url = 'https://api.pushshift.io/reddit/search/submission/?subreddit='
    
    # instantiate empty list    
    df_list = []
    
    # save current epoch, used to iterate in reverse through time
    current_time = epoch_right_now
    
    # set up for loop
    for post in range(n_iter):
        res = requests.get(     
        # instantiate get request
            base_url,
            # requests.get takes base_url and params
            params = {
            # parameters for get request
                
                # specify subreddit
                'subreddit': subreddit,
                # specify number of posts to pull
                'size': 100,
                # ???
                'lang': True,
                # pull everything from current time backward
                'before': current_time}
        )
        
        
        # take data from most recent request, store as df
        df = pd.DataFrame(res.json()['data'])
        # pull specific columns from dataframe for analysis
        df = df.loc[:,[
                        'title',
                        'author',
                        'selftext',
                        'created_utc',
                        'subreddit']]
        
        # append to empty dataframe list
        df_list.append(df)
        
        #add wait time
        time.sleep(20)
        
        # set current time counter back to last epoch in recently grabbed df
        current_time = df['created_utc'].min()
    # return one dataframe for all requests
    return pd.concat(df_list,axis=0)
# Adapated from Tim Book's Lesson Example


# **Use function on WoW subreddit**

# In[22]:


suicide_watch_15k = get_posts('suicidewatch',150,1602714005)


# In[ ]:


suicide_watch_15k.to_csv('./suicide_watch_15k.csv')


# **Combine the DataFrames**

# In[26]:


# both = pd.concat([wow_500,diablo_500])
# both['subreddit'].value_counts()


# In[ ]:


# both.to_csv('./bothredscrape')


# **Note: you can automate this to function to run as a script**
# 
# 1. Add in time breaks between loops to not overload the API
# 2. Convert to .py
# 3. Utilize `caffienate` on terminal
