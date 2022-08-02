#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install opencv-python')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install numpy')


# In[5]:


get_ipython().system('pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html')
get_ipython().system('pip install easyocr')


# In[1]:


import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:



im= './images/car1.jpg'
im


# In[6]:


def recognize_text(im):
    '''loads an image and recognizes text.'''
    
    reader = easyocr.Reader(['en'])
    return reader.readtext(im)


# In[7]:


result = recognize_text(im)


# In[ ]:


result


# In[ ]:


im = cv2.imread(im)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
plt.imshow(im)


# In[ ]:


def overlay_ocr_text(im, save_name):
    '''loads an image, recognizes text, and overlays the text on the image.'''
    
    # loads image
    img = cv2.imread(im)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    dpi = 80
    fig_width, fig_height = int(img.shape[0]/dpi), int(img.shape[1]/dpi)
    plt.figure()
    f, axarr = plt.subplots(1,2, figsize=(fig_width, fig_height)) 
    axarr[0].imshow(img)


# In[ ]:





# In[ ]:


# recognize text
result = recognize_text(im)

    # if OCR prob is over 0.5, overlay bounding box and text
for (bbox, text, prob) in result:
 if prob >= 0.5:
            # display 
  print(f'Detected text: {text} (Probability: {prob:.2f})')

            # get top-left and bottom-right bbox vertices
  (top_left, top_right, bottom_right, bottom_left) = bbox
  top_left = (int(top_left[0]), int(top_left[1]))
  bottom_right = (int(bottom_right[0]), int(bottom_right[1]))

            # create a rectangle for bbox display
  cv2.rectangle(img=img, pt1=top_left, pt2=bottom_right, color=(255, 0, 0), thickness=10)

            # put recognized text
  cv2.putText(img=img, text=text, org=(top_left[0], top_left[1] - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=8)


# In[2]:


pip install pandas


# In[ ]:


import os
path=os.getcwd()
print(path)


# In[ ]:


import pandas as pd


# In[3]:


im3= './images/car5.jpeg'
def recognize_text1(im3):
    '''loads an image and recognizes text.'''
    
    reader = easyocr.Reader(['en'])
    return reader.readtext(im3)
result1 = recognize_text1(im3)
result


# In[ ]:


def overlay_ocr_text(im3, save_name):
    '''loads an image, recognizes text, and overlays the text on the image.'''
    
    # loads image
    img = cv2.imread(im3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    dpi = 80
    fig_width, fig_height = int(img.shape[0]/dpi), int(img.shape[1]/dpi)
    plt.figure()
    f, axarr = plt.subplots(1,2, figsize=(fig_width, fig_height)) 
    axarr[0].imshow(img)


# In[ ]:


im4 = cv2.imread(im4)
im4 = cv2.cvtColor(im4, cv2.COLOR_BGR2RGB)
plt.imshow(im4)


# In[ ]:


def overlay_ocr_text(im4, save_name):
    '''loads an image, recognizes text, and overlays the text on the image.'''
    
    # loads image
    img = cv2.imread(im4)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    dpi = 80
    fig_width, fig_height = int(img.shape[0]/dpi), int(img.shape[1]/dpi)
    plt.figure()
    f, axarr = plt.subplots(1,2, figsize=(fig_width, fig_height)) 
    axarr[0].imshow(img)


# In[ ]:


# recognize text
result = recognize_text(im4)

    # if OCR prob is over 0.5, overlay bounding box and text
for (bbox, text, prob) in result:
 if prob >= 0.5:
            # display 
  print(f'Detected text: {text} (Probability: {prob:.2f})')

            # get top-left and bottom-right bbox vertices
  (top_left, top_right, bottom_right, bottom_left) = bbox
  top_left = (int(top_left[0]), int(top_left[1]))
  bottom_right = (int(bottom_right[0]), int(bottom_right[1]))

            # create a rectangle for bbox display
  cv2.rectangle(img=img, pt1=top_left, pt2=bottom_right, color=(255, 0, 0), thickness=10)

            # put recognized text
  cv2.putText(img=img, text=text, org=(top_left[0], top_left[1] - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=8)


# In[8]:


im5= './images/car8.jpeg'
def recognize_text2(im5):
    '''loads an image and recognizes text.'''
    
    reader = easyocr.Reader(['en'])
    return reader.readtext(im5)
result1 = recognize_text2(im5)
result


# In[ ]:


def overlay_ocr_text(im5, save_name):
    '''loads an image, recognizes text, and overlays the text on the image.'''
    
    # loads image
    img = cv2.imread(im5)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    dpi = 80
    fig_width, fig_height = int(img.shape[0]/dpi), int(img.shape[1]/dpi)
    plt.figure()
    f, axarr = plt.subplots(1,2, figsize=(fig_width, fig_height)) 
    axarr[0].imshow(img)


# In[ ]:


im5 = cv2.imread(im5)
im5 = cv2.cvtColor(im5, cv2.COLOR_BGR2RGB)
plt.imshow(im5)


# In[ ]:


def overlay_ocr_text(im5, save_name):
    '''loads an image, recognizes text, and overlays the text on the image.'''
    
    # loads image
    img = cv2.imread(im5)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    dpi = 80
    fig_width, fig_height = int(img.shape[0]/dpi), int(img.shape[1]/dpi)
    plt.figure()
    f, axarr = plt.subplots(1,2, figsize=(fig_width, fig_height)) 
    axarr[0].imshow(img)


# In[ ]:


# recognize text
result = recognize_text(im5)

    # if OCR prob is over 0.5, overlay bounding box and text
for (bbox, text, prob) in result:
 if prob >= 0.5:
            # display 
  print(f'Detected text: {text} (Probability: {prob:.2f})')

            # get top-left and bottom-right bbox vertices
  (top_left, top_right, bottom_right, bottom_left) = bbox
  top_left = (int(top_left[0]), int(top_left[1]))
  bottom_right = (int(bottom_right[0]), int(bottom_right[1]))

            # create a rectangle for bbox display
  cv2.rectangle(img=img, pt1=top_left, pt2=bottom_right, color=(255, 0, 0), thickness=10)

            # put recognized text
  cv2.putText(img=img, text=text, org=(top_left[0], top_left[1] - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=8)


# In[ ]:


df = pd.read_csv("license plate-data.csv")
print(df.head(25))


# In[ ]:



pip install sklearn


# In[ ]:


text


# In[ ]:


length=len(df)
print(length)


# In[ ]:


df[:length]


# In[ ]:


for i,row in df.iterrows():
    if text in row['LICENSE NO']:
        print(row['NAME'],row['PHONE NO'])
     


# In[ ]:


print("Hi"+NAME)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




