# %%
#Update BfSoup4, import modules and mount the drive to the runtime
#!pip install requests beautifulsoup4 update
import requests, bs4, csv, timeit, datetime
import pandas as pd
import time
from datetime import date

# %%
#from requests.models import encode_multipart_formdata
def frontpage_scraper(filename, adress, pages):
  fieldnames = ["title", "date_created", "OP", "comments", "views", "date_edited", "last_user", "link"]

  with open(filename + ".csv", "w", encoding= "utf-8") as fp:
    writer = csv.DictWriter(fp, fieldnames=fieldnames)
    writer.writeheader()

    for page in range(pages):
      time.sleep(0.2)
      response = requests.get(adress + str(page+1))
      response.encoding = "utf-8"
      response.raise_for_status()
      soup = bs4.BeautifulSoup(response.text, "html.parser")
      for thread in soup.find_all("tr"): #Runs trough all table rows (tr) of the forum front page
        try:
          title = thread.find("div", "title").text
          date_created = thread.find("div", "date").text
          op = thread.find("div", "date_details").text
          comments = thread.find("div", "replies").text
          views = thread.find("div", "views").text
          date_edited = thread.find("div", "last_post").text
          last_user = thread.find("div", "last_post_details").text
          link = thread.find(class_="link",href = True)
                  
          #Writes all the relevant features to the file
          writer.writerow({"title": title, "date_created": date_created, "OP": op.lstrip("af "),\
                           "comments": comments, "views": views, "date_edited": date_edited,\
                           "last_user": last_user.lstrip("af "), "link": link["href"]})
        
        #prevents errors from stopping runtime
        except TypeError:
          continue
        except AttributeError:
          continue

# %%
#Initiate the frontpage_scraper - Scrapes the frontpage of the off-topic forum:
frontpage_scraper("Pokernet_forumindex", "https://www.pokernet.dk/forum/kategorier/frontpage/off-topic.html?p=", 1407)


