#!/usr/bin/env python
# coding: utf-8

# In[37]:


# A new data analyst intern are in charge of monitoring the number of defective products from a specific factory
# He has been told that the number of defects on a given day follows the Poisson distribution with the rate parameter (lambda) 
# You will help him to investigate certain attributes of the Poisson(lambda) distribution to get an intuition for how many defective objects he should expect to see in a given amount of time


# In[ ]:


import scipy.stats as stats
import numpy as np


# In[6]:


# Create a variable called lam that represents the rate parameter
lam = 7 
# 7 defect product each day


# In[7]:


# Calculate and print the probability of observing exactly lam defects on a given day
print(stats.poisson.pmf(lam,lam))


# In[8]:


# Our manager said that having 4 or fewer defects on a given day is an exceptionally good day
# The probability that these events will be happening
print(stats.poisson.cdf(4,lam))


# In[9]:


# On the other hand, having more than 9 defects on any given day is considered a bad day
# The probability that these events will be happening
print(1 - stats.poisson.cdf(9,lam))


# In[10]:


# Simulate 365 days worth of data
year_defect = stats.poisson.rvs(lam, size = 365)
print(year_defect)


# In[12]:


# Print the first 20 values in this data set
print(year_defect[0:20])


# In[14]:


# If we expect 7 defects on a given day, the total number of defects we would expect over 365 days is
print(lam*365)


# In[16]:


# the total sum of the data set year_defects
print(year_defect.sum()) 
# or
print(sum(year_defect))


# In[ ]:


# The total sum of defect in year defect is almost the same as the total number of defect we expect by manually calculating


# In[18]:


# Calculate the average number of defects
print(year_defect.mean())
# nearly the same as the expected number lambda


# In[24]:


# the highest amount of defects in a single day
max_def = year_defect.max()
print(max_def)


# In[25]:


# Calculate and print the probability of observing that maximum value or more
print(1 - stats.poisson.cdf(max_def-1,lam))


# In[30]:


# the number of defects that would put us in the 90th percentile for a given day 3 
stats.poisson.ppf(0.9, lam)


# In[29]:


# There will be about 90% of defects is lower than 10 in a given day
# We can check again and see that the number of defects is lower than 10 occupy 90%
print(stats.poisson.cdf(10, lam))


# In[36]:


# let’s see what proportion of the simulated dataset year_defects
# Count the number of values in the dataset that are greater than or equal to the 90th percentile value
sum(year_defect > 10)/len(year_defect)


# In[ ]:


# the proportion of days that have more than or equal 10 defects is 18% in a year ~ 365

