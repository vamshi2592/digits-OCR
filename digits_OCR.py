
# coding: utf-8

# In[6]:


import matplotlib.pyplot as plt


# In[7]:


from sklearn import datasets, svm, metrics


# In[8]:


digits = datasets.load_digits()


# In[11]:


digits


# In[12]:


digits.data


# In[52]:


digits.target[150]


# In[29]:


digits.images[150]


# In[35]:


list(zip(digits.images, digits.target))


# In[36]:


images_and_labels = list(zip(digits.images, digits.target))


# In[37]:


for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)


# In[38]:


n_samples = len(digits.images)


# In[41]:


n_samples


# In[39]:


data = digits.images.reshape((n_samples, -1))


# In[43]:


data[150]


# In[44]:


classifier = svm.SVC(gamma=0.001)


# In[45]:


classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])


# In[46]:


predicted = classifier.predict(data[n_samples // 2:])


# In[47]:


test_labels = digits.target[n_samples // 2:]


# In[48]:


rep = metrics.classification_report(test_labels, predicted)


# In[49]:


rep


# In[51]:


rep1 = metrics.confusion_matrix(test_labels, predicted)
rep1


# In[60]:


test_digit = ([  0.,   0.,   2.,  12.,   4.,   0.,   0.,   0.,   0.,   1.,  12.,
        16.,  16.,   3.,   0.,   0.,   0.,   7.,  16.,   6.,   4.,  13.,
         0.,   0.,   0.,   8.,  16.,   6.,   0.,  13.,   5.,   0.,   0.,
         1.,  16.,   5.,   0.,   7.,   9.,   0.,   0.,   0.,  16.,   8.,
         0.,   8.,  12.,   0.,   0.,   0.,  13.,  14.,  14.,  16.,  10.,
         0.,   0.,   0.,   4.,  14.,  15.,   7.,   0.,   0.], 
              [  0.,   0.,   2.,  12.,   4.,   0.,   0.,   0.,   0.,   1.,  12.,
        16.,  16.,   3.,   0.,   0.,   0.,   7.,  16.,   6.,   4.,  13.,
         0.,   0.,   0.,   8.,  16.,   6.,   0.,  13.,   5.,   0.,   0.,
         1.,  16.,   5.,   0.,   7.,   9.,   0.,   0.,   0.,  16.,   8.,
         0.,   8.,  12.,   0.,   0.,   0.,  13.,  14.,  14.,  16.,  10.,
         0.,   0.,   0.,   4.,  14.,  15.,   7.,   0.,   0.])


# In[61]:


predict_digit = classifier.predict(test_digit)


# In[62]:


predict_digit


# In[ ]:


images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()

