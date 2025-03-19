import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import spacy
import openai

# Load dataset
file_path = "test.json"
with open(file_path, "r") as file:
    data = json.load(file)

# Convert to DataFrame
df = pd.DataFrame(data)

# Count unique recipes
num_recipes = df.shape[0]

# Flatten ingredient lists
all_ingredients = [ingredient for recipe in df["ingredients"] for ingredient in recipe]

# Count unique ingredients
unique_ingredients = set(all_ingredients)
num_unique_ingredients = len(unique_ingredients)

# Compute ingredient frequency
ingredient_counts = Counter(all_ingredients)
common_ingredients = ingredient_counts.most_common(20)  # Top 20 ingredients

# Plot ingredient distribution
plt.figure(figsize=(12, 6))
plt.barh([x[0] for x in reversed(common_ingredients)], [x[1] for x in reversed(common_ingredients)])
plt.xlabel("Frequency")
plt.ylabel("Ingredient")
plt.title("Top 20 Most Common Ingredients")
plt.show()

# Output summary statistics
num_recipes, num_unique_ingredients, common_ingredients


##############################################


# Co-occurrence patterns
top_ingredients = [item[0] for item in common_ingredients]
co_occurrence_matrix = pd.DataFrame(0, index=top_ingredients, columns=top_ingredients)

# Populate the co-occurrence matrix
for ingredients in df["ingredients"]:
    for ing1 in ingredients:
        for ing2 in ingredients:
            if ing1 in top_ingredients and ing2 in top_ingredients and ing1 != ing2:
                co_occurrence_matrix.loc[ing1, ing2] += 1

# Convert to numpy array for visualization
co_occurrence_array = co_occurrence_matrix.to_numpy()

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(co_occurrence_matrix, annot=False, cmap="Blues", xticklabels=top_ingredients, yticklabels=top_ingredients)
plt.title("Ingredient Co-Occurrence Heatmap (Top 20 Ingredients)")
plt.xlabel("Ingredient")
plt.ylabel("Ingredient")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()


##############################################


# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Use spaCy's nlp.pipe() for batch processing (MUCH FASTER)
docs = list(nlp.pipe(all_ingredients, disable=["ner", "parser"]))  # Disable unnecessary components

# Processed ingredients
tokenized_ingredients = [" ".join([token.lemma_ for token in doc if not token.is_punct and not token.is_stop]) for doc in docs]

# Reshape back into original list-of-lists format
tokenized_index = 0
tokenized_recipe_ingredients = []
for ingredients in df["ingredients"]:
    num_ingredients = len(ingredients)
    tokenized_recipe_ingredients.append(tokenized_ingredients[tokenized_index : tokenized_index + num_ingredients])
    tokenized_index += num_ingredients

# Add the tokenized ingredients back to the DataFrame
df["tokenized_ingredients"] = tokenized_recipe_ingredients

# Show sample results
print(df[["ingredients", "tokenized_ingredients"]].head(10))


##############################################


# Sample function to format data for fine-tuning
def format_for_finetuning(df, output_file="recipes.jsonl"):
    formatted_data = []
    
    for _, row in df.iterrows():
        # Create user prompt based on ingredients
        user_prompt = f"Generate a recipe using the following ingredients: {', '.join(row['ingredients'])}."
        
        # Placeholder for completion (later can use real recipe steps if available)
        assistant_response = "Here’s a recipe:\n1. Mix ingredients...\n2. Cook as needed...\n3. Serve and enjoy!"

        # Structure for OpenAI fine-tuning
        entry = {
            "messages": [
                {"role": "system", "content": "You are a recipe assistant."},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response}
            ]
        }
        formatted_data.append(entry)

    # Save to JSONL file
    with open(output_file, "w") as f:
        for entry in formatted_data:
            f.write(json.dumps(entry) + "\n")

    return output_file

# Run formatting function
formatted_file = format_for_finetuning(df)

