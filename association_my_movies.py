#!/usr/bin/env python
# coding: utf-8

# In[2]:


conda install -c conda-forge mlxtend


# In[16]:


import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder


# In[5]:


movies = pd.read_csv("C:/Users/prate/Downloads/Assignment/Associations/my_movies.csv")
movies.head()


# In[8]:


movies.fillna(0)


# # Pre-Processing
# As the data is not in transaction formation 
# We are using transaction Encoder

# In[9]:


df=pd.get_dummies(movies)
df.head()


# # Apriori Algorithm 

# In[10]:


frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
frequent_itemsets


# In[11]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.7)
rules


# In[7]:


#####  An leverage value of 0 indicates independence. Range will be [-1 1]
# high conviction value means that the consequent is highly depending on the antecedent and range [0 inf]


# In[12]:


rules.sort_values('lift',ascending = False)[0:20]


# In[13]:


rules[rules.lift>1]


# In[17]:


plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


# In[18]:


frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True) #min_support=.5
frequent_itemsets


# In[19]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.7)
rules


# In[20]:


rules.sort_values('lift',ascending = False)[0:20]


# In[21]:


plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


# In[ ]:




