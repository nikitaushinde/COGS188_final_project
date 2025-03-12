from openai import OpenAI
import openai
import json
import random
import pandas as pd


client = OpenAI(
 api_key="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"
)
openai.api_key = "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"


# Load dataset from local JSON file
with open("recipes_raw_nosource_epi.json", "r", encoding="utf-8") as file:
   recipe_data = json.load(file)


# Convert JSON data to DataFrame
recipe_list = []
for key, value in recipe_data.items():
   recipe_list.append({
       "name": value.get("title", "Unknown Recipe"),
       "ingredients": value.get("ingredients", []),
       "instructions": value.get("instructions", "No instructions available")
   })
recipe_df = pd.DataFrame(recipe_list)


def find_recipes(ingredients):
   """Filters recipes that contain the provided ingredients."""
   return recipe_df[recipe_df['ingredients'].apply(lambda x: all(ing in ' '.join(x) for ing in ingredients))]


def generate_recipe(ingredients, kitchen_tools, time_limit):
   """Generates a recipe based on given ingredients, tools, and time limit."""
   available_recipes = find_recipes(ingredients)
  
   if available_recipes.empty:
       return "No recipes found for the given ingredients."
  
   chosen_recipe = available_recipes.sample(1).iloc[0]
  
   # henerate detailed recipe
   prompt = f"Generate a recipe for {chosen_recipe['name']} using {', '.join(ingredients)}. Ensure it can be cooked with {', '.join(kitchen_tools)} within {time_limit} minutes. Provide step-by-step instructions."
   response = openai.ChatCompletion.create(
       model="gpt-3.5-turbo",
       messages=[{"role": "system", "content": "You are a recipe generator AI."},
                 {"role": "user", "content": prompt}]
   )
  
   return response['choices'][0]['message']['content']


# Example usage
ingredients = ["chicken", "tomato"]
kitchen_tools = ["stove", "pan"]
time_limit = 30


recipe = generate_recipe(ingredients, kitchen_tools, time_limit)
print(recipe)
