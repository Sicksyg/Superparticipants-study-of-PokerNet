# %% [markdown]
# **Beautifulsoup method:**

# %%
import requests, bs4, csv, timeit, datetime
import pandas as pd
from datetime import date
import time

# %%
def csv_to_dataframe(data_path):
  with open(data_path, "r", encoding= "utf-8") as scraped:
    df = pd.read_csv(scraped)
    df["date_created"] = pd.to_datetime(df["date_created"], dayfirst=True)
    df["date_edited"] = pd.to_datetime(df["date_edited"], dayfirst=True)
    df["comments"] = df["comments"].astype("int64")
    df["views"] = df["views"].astype("int64")
    
    return df

# %%
scraped_frontpage_df = csv_to_dataframe("Pokernet_forumindex.csv")

# %%
from operator import indexOf
from requests.models import iter_slices

def post_scraper(df, lower_post_thresh, upper_post_thresh, year_thresh):
  '''scrapes all comments from all threads based on the links of the specified dataframe (df)
      df = dataframe with a "link" column
      lower_post_thresh = integer of lowest amount of post comments included in sample
      upper_post_thresh = integer of highest amount of post comments included in sample'''

  tmp_df = df.loc[df["date_created"] >= year_thresh]
  tmp_df = tmp_df.loc[df["comments"] >= lower_post_thresh]
  tmp_df = tmp_df.loc[tmp_df["comments"] < upper_post_thresh]

  '''tmp_df is a temporary dataframe including only the comments within the threshold'''
  
  #creation of the dict with all the data
  all_data = {"thread": [], "thread_created": [], "user": [], "user_link":[], "user_created": [], "user_post_amount": [], "comment_text": [], "meta_date_time": [], "meta_rank_order" : [], "meta_likes": []}
  
  #loops through all indices of the dataframe
  for i in range(len(tmp_df.index)):
    time.sleep(0.1)
    try:
      #response makes requests for each link
      response = requests.get("https://www.pokernet.dk/" + tmp_df["link"].iloc[i]) 
      
      response.encoding = "utf-8"
      response.raise_for_status()
      soup = bs4.BeautifulSoup(response.text, "html.parser")
      thread_name = soup.find("h1", class_="headline").text

      for post in soup.find_all(class_ = "forum_post"):
      
      #finds the elements by their classes
        try:
          if post.find("div", class_="forum_post_pokernet") is not None:
            continue

          user = post.find("div", class_="name").text.strip() 
          
          userlink = post.find("a", itemprop="url", href=True)
          userlink = str(userlink).split()

          user_metadata = post.find(class_= "metadata").text.split()
          
          comment_header = post.find("div", class_="forum_post_header").text.split()
          comment_header[2] = comment_header[2].strip("|")
          
          #Fills the rows with the information from the posts
          all_data["thread"].append(thread_name) #adds the name of the thread
          all_data["thread_created"].append(tmp_df["date_created"].iloc[i])
          all_data["user"].append(user) #adds the user name
          all_data["user_created"].append(user_metadata[1]) #adds the year the user profile was created
          all_data["user_post_amount"].append(user_metadata[3]) #adds how many posts the user has made in total
          all_data["comment_text"].append(post.find("div", class_="post_text").text.strip()) #adds the comment text
          all_data["meta_date_time"].append(pd.to_datetime(comment_header[0] + " " + comment_header[1])) 
          all_data["meta_rank_order"].append(comment_header[2][1:])
          all_data["meta_likes"].append(comment_header[3])
          
          if comment_header[2] == "#1": #makes sure the first link doesn't fuck up everything and adds OP to the username :-)
            all_data["user_link"].append("https://www.pokernet.dk" + userlink[1][6:-1])
            
          else:
            all_data["user_link"].append("https://www.pokernet.dk" + userlink[2][6:-1])
      
        except IndexError:
            return all_data
        except AttributeError:
            return all_data
    except IndexError:
      return all_data
  return all_data

# %%
#Initiate function -  post_scraper
index_df= post_scraper(scraped_frontpage_df, 13, 1000, "04-12-2018")

# %%
#Set up dataframe
pd.set_option('display.max_colwidth', 100)
df1 = pd.DataFrame.from_dict(index_df, orient = "index")
df1 = df1.transpose()

# %%
df1["meta_rank_order"] = df1["meta_rank_order"].astype("int64")
df1["meta_likes"] = df1["meta_likes"].astype("int64")
df1["user_post_amount"] = df1["user_post_amount"].astype("int64")
df1["user_created"] = df1["user_created"].astype("int64")
df1["meta_date_time"] = pd.to_datetime(df1["meta_date_time"])

# %%
df1

# %%
df1.to_csv("Masterfile.csv")


