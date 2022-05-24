# %%
import requests, bs4, csv, timeit, datetime
import pandas as pd

#Datetime to change the date to a datetime
from datetime import date

# Imports of matplot and seaborn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
sns.set(rc={'figure.figsize':(30,20)})

# %% [markdown]
# ### Behandling af  "Masterfile" fra forumscraperen:

# %%
#Mapping funktioner
def remove_op(x):
    if x[-2:] == "OP":
        return x[:-3]
    else:
        return x


# %%
df1 = pd.read_csv('Masterfile.csv', index_col=0)
df1["thread_created"] = pd.to_datetime(df1["thread_created"], dayfirst=True)
df1["meta_date_time"] = pd.to_datetime(df1["meta_date_time"], dayfirst=True)
df1["user"] = df1["user"].map(remove_op)
#df1["isna"] = df1["comment_text"].isna()
#df1.loc[df1["isna"] == True]
df1["comment_text"] = df1["comment_text"].fillna("")
df1

# %%


# %%
def is_reply(text):
    reply = text.split(":")
    if reply[0][-6:] == " skrev":
        return reply[0]


# %%
df1["reply"] = df1["comment_text"].map(is_reply)
df1.loc[df1["reply"] != None]


# %%
df1.loc[df1["user"] == "Eilama"]["thread"].value_counts()

# %%
df1

# %%
thread_starters = df1.loc[df1["meta_rank_order"] == 1]["user"].value_counts()
thread_starters = pd.DataFrame(thread_starters).reset_index()
thread_starters.columns = ["user", "total threads created"]
thread_starters.style.hide_index()
thread_starters.loc[thread_starters["total threads created"] >= thread_starters["total threads created"]]


# %%
new_df = df1[["thread", "user", "user_created"]]
header = ["thread", "user", "year", "active_threads"]
active_df = pd.DataFrame(new_df, columns=header)

#How many threads a user appears in

active_df = active_df.groupby(["thread", "user"]).size().reset_index()["user"].value_counts()
active_df = pd.DataFrame(active_df).reset_index()
active_df.columns = ["user", "active threads"]
active_df.style.hide_index()
active_df.loc[active_df["active threads"] >= active_df["active threads"]]

#new_df.loc["t".value_counts()


# %%
dupe = df1.drop_duplicates(subset=["user_link"])
dupe["user_created"].value_counts()

posts_df = df1["user"].value_counts()

posts_df = pd.DataFrame(posts_df).reset_index()
posts_df.columns = ["user", "total posts"]
posts_df.loc[posts_df["total posts"] >= posts_df["total posts"]]
#posts_df
#.style.hide_index()
#df1["user_created"].value_counts()

# %%
year_df = df1[["user", "user_created"]]
year_df["user"] = year_df["user"].drop_duplicates()
year_df = year_df.dropna()
year_df.reset_index(inplace=True, drop="True")
year_df



# %%
calculated_df = posts_df.merge(active_df, how="inner", on="user")
calculated_df["posts per thread"] = (calculated_df["total posts"] / calculated_df["active threads"]).round(decimals=2)

calculated_df = calculated_df.sort_values("total posts", ascending=False)

calculated_df["percentage of total posts"] = (calculated_df["total posts"] / len(df1) * 100).round(decimals=2)#.astype(str) + "%"

calculated_df = calculated_df.merge(year_df, how="inner", on="user")

calculated_df = calculated_df.merge(thread_starters, how="left", on="user").fillna(0)
calculated_df.sort_values(by="total posts", ascending=False)

calculated_df = calculated_df[["user", "user_created", "total posts", "percentage of total posts", "active threads", "posts per thread", "total threads created"]]
calculated_df["total threads created"] = calculated_df["total threads created"].astype("int64")

calculated_df.describe().round(decimals=2)#.to_excel("description_of_users.xlsx")
calculated_df

# %% [markdown]
# **Models:**

# %%


# %%
def amount(x):
    if x >= 100:
        return "100+"
    elif x >= 50:
        return "50-99"
    elif x >= 25:
        return "25-49"
    else:
        return x

# %%
model_df = df1

# %%
model_df["meta_likes"] = model_df["meta_likes"].map(amount)
model_df = df1.groupby(by="meta_likes").count().reset_index()
model_df = model_df.loc[model_df["meta_likes"] != 0]
model_df.plot.bar(x="meta_likes", y="thread_created")
#sns.barplot(x="percentage of total posts", y="user", data=model_df)

# %%
#sns.barplot(x="percentage of total posts", y="user", data=calculated_df)

# %%
#sns.regplot('total posts', 'active threads', data=calculated_df)

# %%
r2_score(calculated_df['total posts'], calculated_df['active threads'])

# %%
unique_users = calculated_df["user"].unique
unique_users

# %% [markdown]
# **Superposter SP1**

# %%
superposters = calculated_df.loc[calculated_df["percentage of total posts"] >= 1.0]
superposters = superposters[["user", "user_created", "total posts", "percentage of total posts"]]
superposters#["user_created"].value_counts()
#superposters.to_excel("superposters_tabel.xlsx")
sp1_sample = ["NanoQ", "pantherdk", "Aurvandil", "Nilsson"]
superposters["SP"] = "SP"
superposters

# %%
for name in sp1_sample:
    print(name)


# %%
sp1_posts = df1[df1["user"].isin(sp1_sample)]
columns = sp1_posts.columns
sp1_threads = pd.DataFrame(columns=columns)

#sp1_threads = pd.concat([sp1_threads, sp1_posts.loc[sp1_posts["user"] == "pantherdk"].sample(n=3)])
for name in sp1_sample:
   sp1_threads = pd.concat([sp1_threads, sp1_posts.loc[sp1_posts["user"] == name].sample(n=2)])
sp1_threads

# %%
sp1_threads_and_posts = df1[df1["thread"].isin(sp1_threads["thread"])]
sp1_threads_and_posts.to_excel("sp1_dataset.xlsx")

# %% [markdown]
# **Agendasetter SP2**

# %%

#calculated_df["total threads created"].quantile(0.98)
agendasetters = calculated_df.loc[calculated_df["total threads created"] >= calculated_df["total threads created"].quantile(0.99)]
agendasetters = agendasetters[["user", "user_created", "total posts", "total threads created"]].sort_values(by="total threads created", ascending=False)
agendasetters.style.hide_index().to_excel("agendasetters tabel.xlsx")
sp2_sample = ["Newtood", "All-Out-Put", "nikzz", "TUD73"]
agendasetters["SP"] = "SP"
agendasetters


# %%
sp2_posts = df1[df1["user"].isin(sp2_sample)]
sp2_posts = sp2_posts.loc[sp2_posts["meta_rank_order"]==1]
columns = sp2_posts.columns
sp2_threads = pd.DataFrame(columns=columns)

#sp1_threads = pd.concat([sp1_threads, sp1_posts.loc[sp1_posts["user"] == "pantherdk"].sample(n=3)])
for name in sp2_sample:
#sp2_posts.loc[sp2_posts["user"] == "All-Out-Put"]
   sp2_threads = pd.concat([sp2_threads, sp2_posts.loc[sp2_posts["user"] == name].sample(n=2)])

sp2_threads

# %%
sp2_threads_and_posts = df1[df1["thread"].isin(sp2_threads["thread"])]
sp2_threads_and_posts.to_excel("sp2_dataset.xlsx")
sp2_threads_and_posts.thread.value_counts()

# %% [markdown]
# **Facilitators SP3**

# %%

#print(calculated_df["posts per thread"].mean())
facilitators = calculated_df.loc[calculated_df["active threads"] >= calculated_df["active threads"].quantile(0.98)]
#print(facilitators["posts per thread"].mean())
facilitators = facilitators.loc[facilitators["posts per thread"] <= facilitators["posts per thread"].mean()]
facilitators = facilitators[["user", "user_created", "total posts", "active threads", "posts per thread"]].sort_values(by="posts per thread", ascending=True)
#facilitators.sort_values(by="active threads", ascending=False).to_excel("facilitators tabel.xlsx")
sp3_sample = ["dankjar", "Hawkeye", "Micebulldogs", "kris_rem"]
facilitators["SP"] = "SP"
facilitators
#facilitators.style.hide_index().to_excel("facilitators tabel.xlsx"

# %%
merged_df = calculated_df.merge(superposters, how="left", on="user")
merged_df = merged_df.merge(agendasetters, how="left", on="user")
merged_df = merged_df.merge(facilitators, how="left", on="user")
merged_df = merged_df[["user", "user_created_x", "total posts_x", "active threads_x", "posts per thread_x", "SP"]]
merged_df["SP"] = merged_df["SP"].fillna("Not SP")

merged_df
#sns.scatterplot(merged_df, x="total posts_x", hue="SP")


#
#merged_df.plot()

# %%
sp3_posts = df1[df1["user"].isin(sp3_sample)]
columns = sp3_posts.columns
sp3_threads = pd.DataFrame(columns=columns)

#sp1_threads = pd.concat([sp1_threads, sp1_posts.loc[sp1_posts["user"] == "pantherdk"].sample(n=3)])
for name in sp3_sample:
#sp2_posts.loc[sp2_posts["user"] == "All-Out-Put"]
   sp3_threads = pd.concat([sp3_threads, sp3_posts.loc[sp3_posts["user"] == name].sample(n=2)])

sp3_threads

# %%
sp3_threads_and_posts = df1[df1["thread"].isin(sp3_threads["thread"])]
sp3_threads_and_posts.to_excel("sp3_dataset.xlsx")
sp3_threads_and_posts

# %%
import torch
from transformers import AutoTokenizer
from model_def import ElectraClassifier

def load_model():
    model_checkpoint = 'Maltehb/aelaectra-danish-electra-small-cased'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    model = ElectraClassifier(model_checkpoint,2)
    model_path = 'pytorch_model.bin'
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    model.eval()

    return(model, tokenizer)

def make_prediction(text):
    
    tokenized_text = tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding='max_length',
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors='pt',
    )
    input_ids = tokenized_text['input_ids']
    attention_masks = tokenized_text['attention_mask']
    logits = model(input_ids,attention_masks)
    
    logit,preds = torch.max(logits, dim=1)
    return(int(preds))



# %%
#model, tokenizer = load_model()

# %%
#df1["recognition"] = df1["comment_text"].map(make_prediction)


