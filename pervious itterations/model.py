
import json
import random
import pandas as pd
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from tqdm import tqdm

# Load dataset from local JSON file
with open("recipes_raw_nosource_epi.json", "r", encoding="utf-8") as file:
    recipe_data = json.load(file)

# Convert JSON data to DataFrame
recipe_list = [
    {
        "name": value.get("title", "Unknown Recipe"),
        "ingredients": value.get("ingredients", []),
        "instructions": value.get("instructions", "No instructions available")
    }
    for value in recipe_data.values()
]
recipe_df = pd.DataFrame(recipe_list)

'''
FILTER INGREDIENTS
'''

# Flter recipes based on input ingredients



'''
COOKING METHODS
'''

# Preprocess dataset
nlp = spacy.load("en_core_web_sm")
def extract_verbs(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if token.pos_ == "VERB"])

tqdm.pandas(desc="Extracting verbs")
recipe_df['verbs'] = recipe_df['instructions'].progress_apply(extract_verbs)

# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words = 'english', 
    max_features = 1000,
    ngram_range = (1,2),
    min_df = 2,
)
print("Applying TF-IDF...")
X = vectorizer.fit_transform(tqdm(recipe_df['verbs'], desc="TF-IDF Processing"))

# K means clustering
num_clusters = 15
kmeans = KMeans(
    n_clusters = num_clusters, 
    init = 'k-means++',
    max_iter = 1000,
    random_state = 42
)
kmeans.fit(X)

top_n = 5
terms = vectorizer.get_feature_names_out()
clusters = {}

print("Extracting top words for each cluster...")
for i in tqdm(range(num_clusters), desc="Processing Clusters"):
   words = [terms[ind] for ind in kmeans.cluster_centers_[i].argsort()[-top_n:]]
   clusters[f"Cluster {i+1}"] = words





'''
TESTING
'''

for cluster, words in clusters.items():
    print(f"{cluster}: {words}")