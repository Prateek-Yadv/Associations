#!/usr/bin/env python
# coding: utf-8

# In[1]:


#conda install -c conda-forge mlxtend


# In[1]:


import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder


# In[2]:


book = pd.read_csv("C:/Users/prate/Downloads/Assignment/Associations/book.csv")
book.head()


# # Apriori Algorithm 

# In[4]:


frequent_itemsets = apriori(book, min_support=0.1, use_colnames=True)
frequent_itemsets


# In[5]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.7)
rules


# In[7]:


#####  An leverage value of 0 indicates independence. Range will be [-1 1]
# high conviction value means that the consequent is highly depending on the antecedent and range [0 inf]


# In[6]:


rules.sort_values('lift',ascending = False)[0:20]


# In[7]:


rules[rules.lift>1]


# In[8]:


import matplotlib.pyplot as plt
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


# In[15]:


frequent_itemsets = apriori(book, min_support=0.2, use_colnames=True)# min_ support=.6
frequent_itemsets


# In[16]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.7)
rules


# In[17]:


import matplotlib.pyplot as plt
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


# In[ ]:




