#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


l1=[1,2,3,4] 


# In[3]:


n1=np.array(l1)


# In[4]:


n1


# In[6]:


type(n1)


# In[5]:


n2=np.array([[1,2,4,5],[6,7,8,9]])


# In[6]:


n2


# In[12]:


import numpy as np
n1=np.zeros((1,2))


# In[13]:


n1


# In[14]:


import numpy as np
n1=np.zeros((5,5))


# In[15]:


n1


# In[16]:


n1=np.full((2,3),10)
n1


# In[17]:


n1=np.arange(10,20)
n1


# In[18]:


n1=np.arange(10,50,5)
n1


# In[19]:


n1=np.random.randint(100,200,5)
n1


# In[31]:


n1=np.array([[1,2,3,3],[6,7,7,8]])
n1


# In[32]:


n1.shape


# In[36]:


n1.shape=(4,2)
n1


# In[20]:


import numpy as np
n1=np.array([10,20,30,40,50])
n2=np.array([40,50,60,70,80])


# In[ ]:





# In[2]:


np.intersect1d(n1,n2)


# In[3]:


np.setdiff1d(n1,n2)


# In[4]:


np.setdiff1d(n2,n1)


# In[22]:


n1=np.random.randint(1,100,10)


# In[ ]:





# In[23]:


n1


# In[24]:


np.mean(n1)


# In[25]:


np.median(n1)


# In[9]:


np.std(n1)


# In[10]:


n1


# In[13]:


np.save('my_array',n1)


# In[14]:


n2=np.load('my_array.npy')


# In[15]:


n2


# In[ ]:





# In[ ]:


# python pandas labirary


# In[26]:


import pandas as pd
s1=pd.Series([1,2,3,4,5])
s1


# In[20]:


type(s1)


# In[21]:


s1=pd.Series([1,2,3,4,5],index=['a','b','c','d','e'])


# In[22]:


s1


# In[26]:


pd.Series({'k1':10,'k2':20,'k3':40})


# In[3]:


import pandas as pd
s1=pd.Series([1,2,3,4,5])


# In[3]:


s1[2]


# In[4]:


s1[:4]


# In[5]:


s1[-3:]


# In[29]:


import pandas as pd
vk=pd.read_csv( 'C:\\Users\\YASH COMPUTER\\Downloads\\Q9_b.csv')


# In[30]:


vk.head() #it gives the first 5 record


# In[11]:


vk.tail() #it gives the last 5 record


# In[31]:


vk.shape #it gives the size of dataset


# In[14]:


vk.describe() #it gives the some information of table=mean,max,min etc


# # .iloc []  method

# In[ ]:





# In[18]:


vk.min()


# # pythn matplotlib

# In[ ]:


#matplotlib is python library used for data visulation.
#you can create a bar plots,scatter plots,histograms and lot more with matplotlib


# In[36]:


import numpy as np
from matplotlib import pyplot as plt


# In[37]:


x=np.arange(1,11)
x


# In[4]:


y=2*x
y


# In[5]:


plt.plot(x,y)


# In[12]:


plt.plot(x,y,color='r',linestyle=':',linewidth=5)
plt.title('x vs y')
plt.xlabel('this is x axis')
plt.ylabel('this is y axis')
plt.show()


# In[20]:


#adding two line in the same plot
x=np.arange(1,11)
y1=2*x
y2=3*x
plt.plot(x,y1,color='r',linestyle=':',linewidth=3)
plt.plot(x,y2,color='g',linestyle='-',linewidth=2)
plt.title('x vs y')
plt.xlabel('this is x axis')
plt.ylabel('this is y axis')
plt.grid(True)
plt.show()


# In[ ]:


#Adding sub plot


# In[21]:


plt.subplot(2,1,1)
plt.plot(x,y1,color='r',linestyle=':',linewidth=3)

plt.subplot(2,1,2)
plt.plot(x,y2,color='g',linestyle='-',linewidth=2)
plt.title('x vs y')
plt.xlabel('this is x axis')
plt.ylabel('this is y axis')
plt.grid(True)
plt.show()


# In[ ]:


#bar plot


# In[3]:


import numpy as np
student={'vishnu': 50, 'yogi':80,'saurabh':70}


# In[24]:


names=list(student.keys())
names


# In[27]:


marks=list(student.values())
marks


# In[31]:


plt.bar(names,marks)
plt.title('distrubtion of students marks')
plt.xlabel('marks of students')
plt.ylabel('names of students')


# In[32]:


#scatter plot


# In[38]:


x=[10,20,30,40,50]
y=[5,7,9,30,45]
z=[30,4,8,56,34]


# In[36]:


plt.scatter(x,y,marker='*',c='r',s=80)
plt.show()


# In[41]:


#Adding two scatter plot
plt.scatter(x,y,marker='*',c='r',s=80)
plt.scatter(x,z,c='y',s=80)
plt.show()


# In[ ]:


# Adding sub plot


# In[44]:


plt.subplot(1,2,1)
plt.scatter(x,y,marker='*',c='r',s=100)

plt.subplot(1,2,2)
plt.scatter(x,z,c='g',s=100)
 


# In[48]:


#histogram
data=[1,3,4,5,4,6,7,8,9]
plt.hist(data,color='orange')
plt.show


# In[ ]:


#boxplot


# In[50]:


one=[1,2,3,4,5,6,7,8,9]
two=[1,2,3,5,7,9]
three=[1,2,3,4,5,6,7,8,9,7]


# In[51]:


data=list([one,two,three])


# In[52]:


data


# In[54]:


plt.boxplot(data)
plt.show()


# In[33]:


plt.violinplot(data) #violinplot
plt.show()


# In[ ]:


#piechart


# In[38]:


fruit=['apple','orange','mango','gauva']
quantity=[67,34,100,29]


# In[39]:


plt.pie(quantity,labels=fruit)
plt.show()


# In[1]:


# plt.pie(quantity,labels=fruit,autopct='%0.1f%%',colors=['yello','orange','red','blue'])
# plt.show()


# In[10]:


#doughNut-chart


# In[11]:


plt.pie(quantity,labels=fruit,radius=2)
plt.pie([100],colors=['w'],radius=1)
plt.show()


# # seaborn line plot

# In[14]:


import seaborn as sns
from matplotlib import pyplot as plt


# In[3]:


# vk=sns.load_dataset("vk")
# vk.head()


# In[ ]:





# In[ ]:





# In[ ]:




