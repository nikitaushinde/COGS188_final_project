import json
import random
import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# NLTK SETUP: Configure directory and resources (using manual directory if needed)
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
    Splits the instructions into sentences based on punctuation and removes duplicate sentences.
    """
    sentences = re.split(r'(?<=[.!?])\s+', instructions)
    seen = set()
    deduped_sentences = []
    for sentence in sentences:
        stripped = sentence.strip()
        if stripped and stripped not in seen:
            deduped_sentences.append(stripped)
            seen.add(stripped)
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
    Deduplicates the instructions before returning.
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

# LEARNING CURVE FUNCTION
def plot_learning_curve(df, vectorizer, num_clusters=15, random_state=42):
    """
    Plots a learning curve by computing the KMeans inertia for various training set sizes.
    """
    fractions = [0.1, 0.3, 0.5, 0.7, 1.0]
    inertias = []
    total_samples = len(df)
    for frac in fractions:
        sample_df = df.sample(frac=frac, random_state=random_state)
        X_sample = vectorizer.fit_transform(sample_df['verbs'])
        model = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=1000, random_state=random_state)
        model.fit(X_sample)
        inertias.append(model.inertia_)
    sample_sizes = [int(frac * total_samples) for frac in fractions]
    plt.plot(sample_sizes, inertias, marker='o')
    plt.xlabel("Number of training samples")
    plt.ylabel("Inertia (Sum of Squared Distances)")
    plt.title("Learning Curve: KMeans Clustering Performance")
    plt.grid(True)
    plt.show()

# VALIDATION CURVE FUNCTION: Hyper-parameter exploration for number of clusters
def plot_validation_curve(X, cluster_range, random_state=42):
    """
    Plots a validation curve by computing KMeans inertia for a range of cluster numbers.
    """
    inertias = []
    for n_clusters in cluster_range:
        model = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=1000, random_state=random_state)
        model.fit(X)
        inertias.append(model.inertia_)
    plt.plot(cluster_range, inertias, marker='o')
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia (Sum of Squared Distances)")
    plt.title("Validation Curve: Inertia vs. Number of Clusters")
    plt.grid(True)
    plt.show()

# EVALUATION METRIC: BLEU scores
def evaluate_deduplication(original, deduped):
    """
    Evaluates deduplication by computing the BLEU score between the original and deduplicated instructions.
    """
    reference = re.split(r'(?<=[.!?])\s+', original.strip())
    candidate = re.split(r'(?<=[.!?])\s+', deduped.strip())
    smoothing = SmoothingFunction().method1
    score = sentence_bleu([reference], candidate, smoothing_function=smoothing)
    return score

def evaluate_generation(df, num_samples=10):
    """
    Computes and prints the average BLEU score for deduplication over a sample of recipes.
    """
    scores = []
    for i in range(num_samples):
        selected = df.sample(1).iloc[0]
        original = selected['instructions']
        deduped = deduplicate_instructions(original)
        score = evaluate_deduplication(original, deduped)
        scores.append(score)
    avg_score = sum(scores) / len(scores)
    print(f"Average BLEU score for deduplication over {num_samples} samples: {avg_score:.4f}")
    return avg_score

# USER INTERFACE 
def main():
    print("Welcome to PantryPal!")
    while True:
        print("\nPlease choose an option:")
        print("1. Filter recipes by key ingredients (include/exclude)")
        print("2. Generate a recipe based on key ingredients (include/exclude)")
        print("3. Evaluate deduplication quality (BLEU score)")
        print("4. Plot learning curve for KMeans clustering")
        print("5. Plot validation curve (exploring number of clusters)")
        print("6. Exit")
        choice = input("Enter your choice (1/2/3/4/5/6): ").strip()
        
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
            print("\nEvaluating deduplication quality on a sample of recipes...\n")
            evaluate_generation(recipe_df, num_samples=10)
        
        elif choice == '4':
            print("\nPlotting learning curve for KMeans clustering...\n")
            plot_learning_curve(recipe_df, vectorizer, num_clusters=15, random_state=42)
        
        elif choice == '5':
            print("\nPlotting validation curve for different numbers of clusters...\n")
            cluster_range = range(5, 31, 5)  # For example, 5, 10, 15, 20, 25, 30 clusters
            plot_validation_curve(X, cluster_range, random_state=42)
        
        elif choice == '6':
            print("Thank you for using PantryPal. Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
