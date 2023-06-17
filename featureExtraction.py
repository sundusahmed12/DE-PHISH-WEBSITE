#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing required packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 


# In[2]:


#loading the phishing URLs data
data0 = pd.read_csv('URLdata.csv')
data0.head() 


# In[3]:


data0.rename(columns={"domain": "url"}, inplace=True) 


# In[4]:


data0.head() 


# In[5]:


#Collecting 5,000 Phishing URLs randomly
phishurl = data0.sample(n = 5000, random_state = 12).copy()
phishurl = phishurl.reset_index(drop=True)
phishurl.head() 


# In[6]:


phishurl.shape 


# In[7]:


#loading the Legitimate URLs data
data1 = pd.read_csv("URLdata.csv")
data1.head() 


# In[8]:


data1.rename(columns={"domain": "URLs"}, inplace=True) 


# In[9]:


data1.head() 


# In[10]:


#Collecting 5,000 Legitimate URLs randomly
legiurl = data1.sample(n = 5000, random_state = 12).copy()
legiurl = legiurl.reset_index(drop=True)
legiurl.head() 


# In[11]:


legiurl.shape 


# In[12]:


# importing required packages for Feature Extraction
from urllib.parse import urlparse,urlencode
import ipaddress
import re 


# In[13]:


# 1.Domain of the URL (Domain) 
def getDomain(url):  
  domain = urlparse(url).netloc
  if re.match(r"^www.",domain):
	       domain = domain.replace("www.","")
  return domain 


# In[14]:


# 2. IP Address in the URL Feature
def havingIP(url):
  try:
    ipaddress.ip_address(url)
    ip = 1
  except:
    ip = 0
  return ip 


# In[15]:


# 3. "@" Symbol in URL Feature
def haveAtSign(url):
  if "@" in url:
    at = 1    
  else:
    at = 0    
  return at 


# In[16]:


# 4. Length of URL Feature
def getLength(url):
  if len(url) < 100:
    length = 0            
  else:
    length = 1            
  return length 


# In[17]:


# 5. Depth of URL Feature
def getDepth(url):
  s = urlparse(url).path.split('/')
  depth = 0
  for j in range(len(s)):
    if len(s[j]) != 0:
      depth = depth+1
  return depth 


# In[18]:


# 6. Redirection "//" in URL Feature
def redirection(url):
  pos = url.rfind('//')
  if pos > 6:
    if pos > 7:
      return 1
    else:
      return 0
  else:
    return 0 


# In[19]:


# 7. "http/https" in Domain name Feature
def httpDomain(url):
  domain = urlparse(url).netloc
  if 'https' in domain:
    return 1
  else:
    return 0 


# In[20]:


#listing shortening services Feature
shortening_services = r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|" \
                      r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|" \
                      r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|" \
                      r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|db\.tt|" \
                      r"qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|q\.gs|is\.gd|" \
                      r"po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|" \
                      r"prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|" \
                      r"tr\.im|link\.zip\.net"  


# In[21]:


# 8. Using URL Shortening Services “TinyURL” Feature
def tinyURL(url):
    match=re.search(shortening_services,url)
    if match:
        return 1
    else:
        return 0 


# In[22]:


# 9. Prefix or Suffix "-" in Domain Feature
def prefixSuffix(url):
    if '-' in urlparse(url).netloc:
        return 1            
    else:
        return 0 


# In[23]:


# get_ipython().system('pip install python-whois')


# In[24]:


# importing required packages for Domain Based Features
import re
from bs4 import BeautifulSoup
import whois
import urllib
import urllib.request
from datetime import datetime  


# In[25]:


# 10.DNS Record availability (DNS_Record) Feature
# obtained in the featureExtraction function itself 


# In[26]:


# 11. Web Traffic Feature
def web_traffic(url):
  try:
    #Filling the whitespaces in the URL if any
    url = urllib.parse.quote(url)
    rank = BeautifulSoup(urllib.request.urlopen("http://data.alexa.com/data?cli=10&dat=s&url=" + url).read(), "xml").find(
        "REACH")['RANK']
    rank = int(rank)
  except TypeError:
        return 1
  if rank <100000:
    return 1
  else:
    return 0 


# In[27]:


# 12. Age of Domain Feature
def domainAge(domain_name):
  creation_date = domain_name.creation_date
  expiration_date = domain_name.expiration_date
  if (isinstance(creation_date,str) or isinstance(expiration_date,str)):
    try:
      creation_date = datetime.strptime(creation_date,'%Y-%m-%d')
      expiration_date = datetime.strptime(expiration_date,"%Y-%m-%d")
    except:
      return 1
  if ((expiration_date is None) or (creation_date is None)):
      return 1
  elif ((type(expiration_date) is list) or (type(creation_date) is list)):
      return 1
  else:
    ageofdomain = abs((expiration_date - creation_date).days)
    if ((ageofdomain/30) < 6):
      age = 1
    else:
      age = 0
  return age 


# In[28]:


# 13. End Period of Domain Feature
def domainEnd(domain_name):
  expiration_date = domain_name.expiration_date
  if isinstance(expiration_date,str):
    try:
      expiration_date = datetime.strptime(expiration_date,"%Y-%m-%d")
    except:
      return 1
  if (expiration_date is None):
      return 1
  elif (type(expiration_date) is list):
      return 1
  else:
    today = datetime.now()
    end = abs((expiration_date - today).days)
    if ((end/30) < 6):
      end = 0
    else:
      end = 1
  return end 


# In[29]:





# In[30]:


# importing required packages for HTML and JavaScript based Features
import requests 


# In[31]:


# 14. IFrame Redirection Feature
def iframe(response):
  if response == "":
      return 1
  else:
      if re.findall(r"[|]", response.text):
          return 0
      else:
          return 1


# In[32]:


# 15. Status Bar Customization Feature
def mouseOver(response): 
  if response == "" :
    return 1
  else:
    if re.findall("", response.text):
      return 1
    else:
      return 0


# In[33]:


# 16. Disabling Right Click Feature
def rightClick(response):
  if response == "":
    return 1
  else:
    if re.findall(r"event.button ?== ?2", response.text):
      return 0
    else:
      return 1 


# In[34]:


# 17. Website Forwarding Feature
def forwarding(response):
  if response == "":
    return 1
  else:
    if len(response.history) <= 2:
      return 0
    else:
      return 1


# In[35]:


#Function to extract features
def featureExtraction(url,label):

  features = []
  #Address bar based features (9)
  features.append(getDomain(url))
  features.append(havingIP(url))
  features.append(haveAtSign(url))
  features.append(getLength(url))
  features.append(getDepth(url))
  features.append(redirection(url))
  features.append(httpDomain(url))
  features.append(tinyURL(url))
  features.append(prefixSuffix(url))
  
  #Domain based features (4)
  dns = 0
  try:
    domain_name = whois.whois(urlparse(url).netloc)
  except:
    dns = 1

  features.append(dns)
  features.append(web_traffic(url))
  features.append(1 if dns == 1 else domainAge(domain_name))
  features.append(1 if dns == 1 else domainEnd(domain_name))
  
  # HTML & Javascript based features (4)
  try:
    response = requests.get(url)
  except:
    response = ""
  features.append(iframe(response))
  features.append(mouseOver(response))
  features.append(rightClick(response))
  features.append(forwarding(response))
  features.append(label)
  
  return features 


# In[36]:


#loading the phishing URLs data
phishing = pd.read_csv('URLdata.csv')
phishing.head() 


# In[49]:


#loading the legitimate URLs data
legitimate = pd.read_csv('URLdata.csv')
legitimate.head() 


# In[50]:


urldata = pd.read_csv('URLdata.csv') 
urldata.head() 


# In[51]:


urldata.tail() 


# In[52]:


urldata.shape 


# In[53]:


# Storing the data in CSV file
urldata.to_csv('URLDATA.csv', index=False) 


# In[ ]:

def generate_data_set(url):
    # Address bar based features (9)
    domain = getDomain(url)
    ip = havingIP(url)
    at = haveAtSign(url)
    length = getLength(url)
    depth = getDepth(url)
    redirect = redirection(url)
    http_domain = httpDomain(url)
    tiny_url = tinyURL(url)
    prefix_suffix = prefixSuffix(url)

    # Domain based features (4)
    dns = 0
    try:
        domain_name = whois.whois(urlparse(url).netloc)
    except:
        dns = 1
    dns_record = dns
    web_traffic = web_traffic(url)
    domain_age = 1 if dns == 1 else domainAge(domain_name)
    domain_end = 1 if dns == 1 else domainEnd(domain_name)

    # HTML & Javascript based features (4)
    try:
        response = requests.get(url)
    except:
        response = ""
    iframe_feature = iframe(response)
    mouse_over = mouseOver(response)
    right_click = rightClick(response)
    web_forwards = forwarding(response)

    # Create the feature list
    features = [
        domain,
        ip,
        at,
        length,
        depth,
        redirect,
        http_domain,
        tiny_url,
        prefix_suffix,
        dns_record,
        web_traffic,
        domain_age,
        domain_end,
        iframe_feature,
        mouse_over,
        right_click,
        web_forwards,
    ]

    return features

