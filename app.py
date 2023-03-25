#!/usr/bin/env python
# coding: utf-8

# In[3]:


#/export
from fastai.vision.all import *
import gradio as gr

def is_cat(x): 
    return x[0].isupper()


# In[4]:


im = PILImage.create('dog.jpg')
im.thumbnail((192, 192))
im


# In[6]:


learn = load_learner('model.pkl')


# In[7]:


learn.predict(im)


# In[8]:


#/export
categories = ('Dog', 'Cat')

def classify_image(img):
  pred, idx, probs = learn.predict(img)
  return dict(zip(categories, map(float, probs)))


# In[9]:


classify_image(im)


# In[11]:


#/export
image = gr.inputs.Image(shape=(192, 192))
label = gr.outputs.Label()
examples = ['dog.jpg', 'cat.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)


# ## export

# In[17]:


# import nbdev
# nbdev.export.nb_export('app.ipynb', 'app')

