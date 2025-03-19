import json
import random
import pandas as pd
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from tqdm import tqdm


# NLTK SETUP: Configure directory and resources (my NLTK would not work, so I had to 
# download it manually and specify the directory)

nltk_data_dir = 'C:/Users/dylan/nltk_data'
nltk.data.path.append(nltk_data_dir)
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('punkt_tab', download_dir=nltk_data_dir)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)

# COMMENT OUT CODE ABOVE AND REPLACE WITH CODE BELOW, IF IT DOES NOT WORK CHANGE 
# THE CODE ABOVE TO FIT YOUR OWN DIRECTORY 

# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# DATA LOADING AND PREPARATION

with open("recipes_raw_nosource_epi.json", "r", encoding="utf-8") as file:
    recipe_data = json.load(file)

recipe_list = [
    {
        "name": value.get("title", "Unknown Recipe"),
        "ingredients": value.get("ingredients", []),
        "instructions": value.get("instructions", "No instructions available")
    }
    for value in recipe_data.values()
]
recipe_df = pd.DataFrame(recipe_list)


# FILTERING FUNCTION: Inclusions and Exclusions

def filter_recipes_by_inclusions_and_exclusions(include, exclude, df):
    """
    Returns a DataFrame of recipes that include all of the ingredients in 'include'
    and do NOT include any ingredients in 'exclude' (case-insensitive).
    """
    include = [item.lower() for item in include]
    exclude = [item.lower() for item in exclude]
    
    def filter_func(ingredients_list):
        combined = " ".join(ingredients_list).lower()
        includes_all = all(ing in combined for ing in include)
        excludes_all = not any(ex in combined for ex in exclude)
        return includes_all and excludes_all
    
    return df[df['ingredients'].apply(filter_func)]

# DEDUPLICATION HELPER FOR INSTRUCTIONS 

def deduplicate_instructions(instructions):
    """
    Splits the instructions into sentences based on punctuation and
    remove duplicates
    """
    # Split by sentence-ending punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', instructions)
    seen = set()
    deduped_sentences = []
    for sentence in sentences:
        stripped = sentence.strip()
        if stripped and stripped not in seen:
            deduped_sentences.append(stripped)
            seen.add(stripped)
    # Rejoin with a space between sentences.
    return " ".join(deduped_sentences)

# COOKING METHODS PROCESSING (Using NLTK)

def extract_verbs(text):
    """
    Tokenizes the text and returns the grouped verbs.
    """
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    verbs = [lemmatizer.lemmatize(word.lower(), 'v') for word, tag in pos_tags if tag.startswith('VB')]
    return ' '.join(verbs)

tqdm.pandas(desc="Extracting verbs")
recipe_df['verbs'] = recipe_df['instructions'].progress_apply(extract_verbs)

# TF-IDF and K-Means Clustering on Extracted Verbs

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=1000,
    ngram_range=(1, 2),
    min_df=2,
)
print("Applying TF-IDF...")
X = vectorizer.fit_transform(tqdm(recipe_df['verbs'], desc="TF-IDF Processing"))

num_clusters = 15
kmeans = KMeans(
    n_clusters=num_clusters,
    init='k-means++',
    max_iter=1000,
    random_state=42
)
kmeans.fit(X)

top_n = 5
terms = vectorizer.get_feature_names_out()
clusters = {}

print("Extracting top words for each cluster...")
for i in tqdm(range(num_clusters), desc="Processing Clusters"):
    words = [terms[ind] for ind in kmeans.cluster_centers_[i].argsort()[-top_n:]]
    clusters[f"Cluster {i+1}"] = words

print("\nCooking Method Clusters:")
for cluster, words in clusters.items():
    print(f"{cluster}: {words}")

# RECIPE GENERATION 

def generate_recipe(include, exclude, df):
    """
    Filters recipes by the given inclusions and exclusions and randomly selects one.
    """
    filtered = filter_recipes_by_inclusions_and_exclusions(include, exclude, df)
    if filtered.empty:
        return "No recipes found with the specified criteria."
    selected = filtered.sample(1).iloc[0]
    clean_instructions = deduplicate_instructions(selected['instructions'])
    recipe_text = (
        f"Recipe Name: {selected['name']}\n\n"
        f"Ingredients:\n- " + "\n- ".join(selected['ingredients']) + "\n\n"
        f"Instructions:\n{clean_instructions}\n"
    )
    return recipe_text

# USER INTERFACE 

def main():
    print("Welcome to PantryPals!")
    while True:
        print("\nPlease choose an option:")
        print("1. Filter recipes by key ingredients (include/exclude)")
        print("2. Generate a recipe based on key ingredients (include/exclude)")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ").strip()
        
        if choice == '1':
            include_input = input("Enter ingredients to include (comma separated): ")
            exclude_input = input("Enter ingredients to exclude (comma separated): ")
            include_list = [i.strip() for i in include_input.split(",") if i.strip()]
            exclude_list = [i.strip() for i in exclude_input.split(",") if i.strip()]
            filtered = filter_recipes_by_inclusions_and_exclusions(include_list, exclude_list, recipe_df)
            if filtered.empty:
                print("No recipes found with those criteria.")
            else:
                print(f"\nFound {len(filtered)} recipe(s):")
                for idx, row in filtered.iterrows():
                    print(f"\nName: {row['name']}")
                    print("Ingredients:")
                    for ing in row['ingredients']:
                        print(f" - {ing}")
                    print("-----")
                    
        elif choice == '2':
            include_input = input("Enter ingredients to include (comma separated): ")
            exclude_input = input("Enter ingredients to exclude (comma separated): ")
            include_list = [i.strip() for i in include_input.split(",") if i.strip()]
            exclude_list = [i.strip() for i in exclude_input.split(",") if i.strip()]
            print("\nGenerating recipe...\n")
            recipe = generate_recipe(include_list, exclude_list, recipe_df)
            print(recipe)
            
        elif choice == '3':
            print("Thank you for using PantryPals. Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
