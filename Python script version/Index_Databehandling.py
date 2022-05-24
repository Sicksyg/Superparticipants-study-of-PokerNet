# %%
import requests, bs4, csv, timeit, datetime
import pandas as pd

#Datetime to change the date to a datetime
from datetime import date

import numpy as np
# Imports of matplot and seaborn
import matplotlib.pyplot as plt
import seaborn as sbn
sbn.set(rc={'figure.figsize':(10,10)})

# %% [markdown]
# ### Indlæsning af "Forumindex" dataframe:

# %%
indexdf = pd.read_csv('Pokernet_forumindex.csv')


# %%


# %%
#Function to find a specific cell based on a column and row
#indexdf["date_created"].iloc[0]

# %% [markdown]
# ### Dataformatering af dataframet:

# %%
#changes date_created and date_edited to datetime type
indexdf["date_created"] = pd.to_datetime(indexdf["date_created"], dayfirst=True)
indexdf["date_edited"] = pd.to_datetime(indexdf["date_edited"], dayfirst=True)
indexdf["comments"] = indexdf["comments"].astype("int64")
indexdf["views"] = indexdf["views"].astype("int64")
indexdf.info()

# %% [markdown]
# ### Deskreptiv statistik af index:

# %%

indexdf.loc[indexdf["title"] == "220V stik i USA?"] # <---- første rigtige Off-Topic opslag
indexdf = indexdf.loc[indexdf["date_created"] > "2008-01-01"] # <---- Her begynder off-topic at være hovedsageligt off-topic fremfor bad beats
indexdf["title"].count()

# %%


# %%
thread_start_df = indexdf.loc[indexdf["date_created"] >= "04-12-2018"]
thread_start_df = thread_start_df.loc[thread_start_df["comments"] >= 13]
thread_start_df = thread_start_df.loc[thread_start_df["comments"] <= 1000]
thread_start_df

threads_started = pd.DataFrame(thread_start_df["OP"].value_counts()).reset_index()
threads_started.columns=["user", "threads started"]
threads_started.head(30)


# %%
#Outliers:
df_comm_outlier = indexdf.loc[indexdf["comments"] < 1000]
df_views_outlier = indexdf.loc[indexdf["views"] < 100000]

# %%
#find outlies med denne funktion

year4df = indexdf.loc[indexdf["date_created"] >= "04-12-2018"]
#year4df = year4df.loc[year4df["comments"] >= 13]
#year4df = year4df.loc[year4df["comments"] < 1000]
indexdf["comments"].corr(indexdf["views"])

#year4df.describe()
#year4df["title"].nunique()
sorted_inplace = year4df.sort_values(by="comments", ignore_index=True, ascending=False)


#df.loc[df["views"] > 100000]

# %%
# Show the spread of comments per post

#sorted_year4df = year4df.sort_values(by="comments")
#sorted_index.groupby()
#sbn.countplot(x = "comments", data=sorted_year4df)
#sbn.lineplot(data=sorted_year4df, x= sorted_year4df.index, y="comments")¨

x = sorted_inplace["comments"]
y = sorted_inplace.index


sbn.set(style="whitegrid")
sbn.scatterplot(x, y, size=x).set(ylabel="threads")

#sbn.lineplot(x, y)

#semilogy

#fig, ax = plt.subplots()

#y_avg = [np.mean(x)] * len(x)
#y_med = [np.median(x)] * len(x)

#ax.plot(x ,lw = 4, label='comments per thread')
#ax.plot(y_avg, x, color='red', lw=2, label="average", linestyle="dashed")
#ax.plot(y_med, x, color="green", lw=2, label="median", linestyle="dashed")
#ax.plot([550, 0], [x.median(), x.median()],label="median")
#ax.plot([362, 0],[x.mean(), x.mean()],  color="red", label="mean")

#plt.legend(fontsize=20)
#plt.xlabel("threads")
#plt.ylabel("comments")

#ax.set_yscale('log')
#ax.set_xscale('log')
#plt.xticks([x.median(), x.median(), 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700])
#plt.show()




#sorted_inplace["comments"].barplot()
#plt.hist("comments", bins=sorted_index["comments"].value_counts)

# %%

#df with a count of threads, posts and views  per year
indexdf["year_created"] = indexdf["date_created"].map(lambda x: x.year)
groupby_index = indexdf.groupby(by="year_created").agg({"title": "count", "comments":"sum", "views":"sum"}).reset_index()

# %%

print(groupby_index)
x = groupby_index["year_created"]
y = groupby_index

figure, axis = plt.subplots(3, 1, figsize=(10,25))

axis[0].plot(x, y["title"],linewidth=2)
axis[0].set_title("Count of threads per year", size=25)
axis[0].set_xlabel('Year', fontsize = 20)
axis[0].set_ylabel('Threads', fontsize = 20)
axis[0].tick_params(axis='both', which='major', labelsize=20)

axis[1].plot(x, y["comments"],linewidth=2)
axis[1].set_title("Count of comments per year", size=25)
axis[1].set_xlabel('Year', fontsize = 20)
axis[1].set_ylabel('Comments', fontsize = 20)
axis[1].tick_params(axis='both', which='major', labelsize=20)

axis[2].plot(x, y["views"],linewidth=2)
axis[2].set_title("Count of views per year", size=25)
axis[2].set_xlabel('Year', fontsize = 20)
axis[2].set_ylabel('Views', fontsize = 20)
axis[2].tick_params(axis='both', which='major', labelsize=20)

plt.ticklabel_format(style = 'plain')

plt.show()

# %%
def is_tilt(a):
    


# %%
#indexdf["title"].str.contains("tilt")
tmp_df = indexdf
searcher = "tilt" #"V2|V3|V4|V5|V6"
tmp_df["is_tilted"] = indexdf["title"].str.contains(searcher)

tmp_df.loc[tmp_df["is_tilted"] == True]

#tiltseries.value_counts()


