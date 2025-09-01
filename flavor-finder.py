from typing import List, Optional, Dict, Any
from langgraph.graph import START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import re
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(override=True)

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Dietary restriction conflicts mapping
DIETARY_CONFLICTS = {
    "vegetarian": {
        "chicken", "beef", "pork", "fish", "salmon", "tuna", "shrimp", "bacon",
        "sausage", "lamb", "turkey", "duck", "seafood", "anchovies", "prosciutto",
        "pancetta", "chicken breast", "ground beef", "steak", "ham", "salami", "pepperoni"
    },
    "vegan": {
        "chicken", "beef", "pork", "fish", "salmon", "tuna", "shrimp", "bacon",
        "sausage", "lamb", "turkey", "duck", "seafood", "milk", "cheese", "butter",
        "eggs", "yogurt", "cream", "honey", "ghee", "paneer", "mozzarella", "parmesan",
        "ricotta", "brie", "gruyere", "feta", "goat cheese", "cream cheese", "sour cream",
        "whey", "casein", "lactose", "anchovies", "prosciutto", "pancetta",
        "chicken breast", "ground beef", "steak", "ham", "salami", "pepperoni"
    },
    "gluten-free": {
        "bread", "pasta", "flour", "wheat", "barley", "rye", "couscous", "crackers",
        "cookies", "cake", "pastry", "bagels", "pretzels", "beer", "soy sauce",
        "teriyaki sauce", "hoisin sauce", "malt", "wheat flour", "all-purpose flour",
        "bread flour", "cake flour", "whole wheat", "durum", "semolina", "spelt",
        "kamut", "bulgur", "farro", "tortillas", "pita", "naan", "baguette", "croissant"
    },
    "dairy-free": {
        "milk", "cheese", "butter", "yogurt", "cream", "ice cream", "sour cream",
        "whey", "casein", "lactose", "ghee", "paneer", "mozzarella", "parmesan",
        "ricotta", "brie", "gruyere", "feta", "goat cheese", "cream cheese",
        "cottage cheese", "mascarpone", "half and half", "condensed milk",
        "evaporated milk", "buttermilk", "kefir"
    },
    "keto": {
        "bread", "pasta", "rice", "potatoes", "sugar", "honey", "flour", "corn",
        "beans", "lentils", "chickpeas", "quinoa", "oats", "barley", "wheat",
        "couscous", "tortillas", "chips", "crackers", "cookies", "cake", "candy",
        "fruit juice", "soda", "beer", "bananas", "grapes", "mangoes", "pineapple",
        "dates", "raisins"
    },
    "paleo": {
        "bread", "pasta", "rice", "cheese", "milk", "yogurt", "beans", "lentils",
        "chickpeas", "peanuts", "soy", "tofu", "sugar", "corn", "wheat", "oats",
        "barley", "quinoa", "couscous", "potatoes", "dairy", "grains", "legumes",
        "processed foods", "refined sugar", "artificial sweeteners"
    }
}

# Invalid ingredients list - expanded to include inappropriate terms
INVALID_INGREDIENTS = {
    # Non-food items
    "shoe", "shoes", "sock", "socks", "shirt", "pants", "clothes", "clothing",
    "paper", "plastic", "metal", "glass", "wood", "rubber", "fabric", "leather",
    "concrete", "cement", "brick", "stone", "rock", "dirt", "mud", "sand",
    "nail", "screw", "bolt", "hammer", "tool", "wire", "cable", "rope",
    "battery", "electronics", "computer", "phone", "keyboard", "mouse",
    "furniture", "chair", "table", "desk", "bed", "sofa", "couch",
    "car", "tire", "engine", "gasoline", "oil", "brake", "wheel",
    "paint", "glue", "adhesive", "tape", "staple", "clip", "pin",
    "pen", "pencil", "eraser", "marker", "crayon", "chalk",
    "book", "magazine", "newspaper", "cardboard", "styrofoam",
    "detergent", "soap", "shampoo", "bleach", "cleaner", "chemical",
    "poison", "toxic", "waste", "garbage", "trash", "refuse",

    # Body parts and bodily fluids (EXPAND THIS SECTION)
    "finger", "toe", "hand", "foot", "arm", "leg", "head", "hair",
    "nail", "tooth", "teeth", "bone", "skin", "blood", "urine", "feces",
    "saliva", "mucus", "vomit", "puke", "snot", "booger",

    # ANATOMICAL TERMS:
    "penis", "vagina", "vulva", "clitoris", "testicle", "testicles",
    "scrotum", "breast", "breasts", "nipple", "nipples", "anus",
    "rectum", "genitals", "genital", "groin", "buttocks", "butt",
    "pubic", "foreskin", "labia", "ovary", "ovaries", "uterus",
    "prostate", "semen", "sperm", "ejaculate", "menstrual", "period",

    # Profanity and inappropriate language (comprehensive list)
    "shit", "piss", "fuck", "cunt", "cock", "dick", "pussy", "ass", "asshole",
    "bitch", "bastard", "damn", "hell", "crap", "poop", "turd", "fart",
    "bollocks", "bugger", "wanker", "tosser", "prick", "twat", "slag",
    "whore", "slut", "nigger", "nigga", "faggot", "fag", "dyke", "kike",
    "spic", "chink", "gook", "wetback", "beaner", "cracker", "honky",
    "retard", "retarded", "spastic", "spaz", "mongoloid", "midget",

    # Drug-related terms
    "cocaine", "heroin", "meth", "marijuana", "weed", "pot", "crack",
    "ecstasy", "lsd", "mushrooms", "pills", "drugs", "narcotic",

    # Weapons and violence
    "gun", "knife", "bomb", "explosive", "weapon", "bullet", "ammunition",
    "grenade", "missile", "sword", "blade", "poison", "venom",

    # Gross or unsanitary items
    "mold", "mould", "fungus", "bacteria", "virus", "germs", "disease",
    "infection", "pus", "rot", "rotten", "decay", "decomposed", "maggot",
    "fly", "cockroach", "rat", "mouse", "pest", "vermin", "parasite",

    # Nonsensical food combinations
    "asdfgh", "qwerty", "zxcvbn", "gibberish", "blahblah", "whatever",
    "nothing", "anything", "everything", "something", "stuff", "things"
}

# Valid food-related terms (to help with validation)
VALID_FOOD_CATEGORIES = {
    "meat", "vegetable", "fruit", "grain", "dairy", "spice", "herb",
    "sauce", "oil", "vinegar", "seasoning", "protein", "carbohydrate",
    "ingredient", "produce", "pantry", "staple", "condiment"
}


class State(BaseModel):
    messages: List[Any] = Field(default_factory=list)
    ingredients: str
    cuisine: str
    time: int
    meal_type: str
    dietary_restrictions: str


class RecipeRequest(BaseModel):
    ingredients: str
    cuisine: str
    time: int
    meal_type: str
    dietary_restrictions: str


class RecipeResponse(BaseModel):
    success: bool
    recipe: str
    error: Optional[str] = None


class VariationsResponse(BaseModel):
    success: bool
    variations: List[str]
    error: Optional[str] = None


class IngredientSuggestionRequest(BaseModel):
    base_ingredients: str
    cuisine: str
    meal_type: str
    dietary_restrictions: str


class IngredientSuggestionResponse(BaseModel):
    success: bool
    suggested_ingredients: List[str]
    error: Optional[str] = None


class ValidateRecipeRequest(BaseModel):
    available_ingredients: List[str]
    cuisine: str
    time: int
    meal_type: str
    dietary_restrictions: str


class ValidateRecipeResponse(BaseModel):
    success: bool
    can_make_recipe: bool
    recipe: Optional[str] = None
    missing_ingredients: Optional[List[str]] = None
    error: Optional[str] = None


class ValidateIngredientsRequest(BaseModel):
    ingredients: str


class ValidateIngredientsResponse(BaseModel):
    success: bool
    valid_ingredients: List[str]
    invalid_ingredients: List[str]
    error: Optional[str] = None


graph_builder = StateGraph(State)
try:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
except Exception as e:
    logger.error(f"Failed to initialize ChatOpenAI: {e}")
    raise


def is_valid_ingredient(ingredient: str) -> bool:
    """Check if an ingredient is valid (food-related and appropriate)"""
    try:
        normalized = ingredient.lower().strip()

        # Remove common punctuation and extra spaces
        normalized = re.sub(r'[^\w\s-]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)

        # Check if empty or too short
        if len(normalized) < 2:
            return False

        # Check if it's just numbers or random characters
        if normalized.isdigit() or len(set(normalized.replace(' ', ''))) < 3:
            return False

        # WHITELIST CHECK FIRST - foods that might contain problematic substrings
        whitelist_foods = {
            # Contains "toe/toes"
            "tomato", "tomatoes", "potato", "potatoes", "sweet potato", "sweet potatoes",
            "fingerling potato", "fingerling potatoes",

            # Contains "ass"
            "bass", "molasses", "sassafras", "grass-fed", "cassava", "cassoulet",
            "assam tea", "assorted", "sassafras tea", "brassica", "massaman curry",

            # Contains "rum"
            "rump", "rump roast", "rump steak", "crumb", "crumbs", "breadcrumb",
            "breadcrumbs", "crumpet", "crumpets", "rum cake", "rum raisin",
            "cumin", "cardamom", "mushroom", "mushrooms", "rumaki",

            # Contains "cock"
            "cocktail", "cocktails", "cockles", "cockerel", "cock-a-leekie",
            "cockscomb", "cocktail sauce", "cocktail onions",

            # Contains "hell"
            "shell", "shells", "shellfish", "hellim", "seashell pasta",
            "soft shell crab", "hard shell crab",

            # Contains "crap"
            "crab", "crabs", "crabmeat", "crabapple", "crab cake", "crab legs",
            "scrapple", "crab rangoon", "crab bisque",

            # Contains "prick"
            "prickly pear", "prickly",

            # Contains "pee"
            "peel", "peeled", "chickpea", "chickpeas", "split pea", "split peas",
            "black-eyed pea", "black-eyed peas", "sweet pea", "snow pea", "snap pea",
            "pea shoots", "pea pods",

            # Contains "tit"
            "petit", "petits", "appetite", "petite", "petit four", "petit pois",

            # Contains "nig"
            "nigiri", "nigiri sushi", "vinegar", "nigella", "nigella seeds",

            # Contains "fag"
            "faggot", "faggots", "fagara", "fagottini",

            # Contains "rape"
            "grape", "grapes", "grapefruit", "rapeseed", "rapeseed oil",
            "canola", "grape leaves", "grape tomato", "grape tomatoes",
            "grapevine", "grape juice", "capers",

            # Contains "arse"
            "coarse", "coarse salt", "coarse pepper",

            # Contains "dick"
            "pickled",

            # Contains "puss"
            "octopus",

            # Contains "dam/damn"
            "edamame", "macadamia", "macadamia nuts", "cardamom", "damson",
            "damson plum",

            # Contains "poo"
            "spoon", "spoons", "teaspoon", "tablespoon", "poolish",

            # Contains "cum"
            "cucumber", "cucumbers", "kumquat", "cumin",

            # Contains "spic"
            "spice", "spices", "spicy", "allspice", "spiced",

            # Other potentially problematic
            "shiitake", "shitake", "urfa", "dongpo",

            # Common ingredients that should always pass
            "salt", "pepper", "oil", "butter", "sugar", "flour", "water",
            "milk", "eggs", "cheese", "bread", "rice", "pasta", "meat",
            "chicken", "beef", "pork", "fish", "vegetables", "fruits",
            "herbs", "sauce", "soup", "salad", "sandwich", "vinegar",
            "garlic", "onion", "onions", "basil", "oregano", "thyme",
            "rosemary", "parsley", "cilantro", "mint", "dill", "sage"
        }

        # Check if ingredient matches whitelist (exact or as part of compound)
        for whitelisted in whitelist_foods:
            if normalized == whitelisted or whitelisted in normalized:
                return True

        # Check for valid food categories
        for valid_category in VALID_FOOD_CATEGORIES:
            if valid_category in normalized:
                return True

        # NOW check invalid ingredients - but only as whole words
        words = normalized.split()
        for word in words:
            if word in INVALID_INGREDIENTS:
                return False

        # Check if the entire normalized string is invalid
        if normalized in INVALID_INGREDIENTS:
            return False

        # For any remaining ingredients, be permissive if they're reasonable length
        if 2 <= len(normalized) <= 40:
            suspicious_patterns = ['www', 'http', '@', '.com', '\\', '/', '<', '>']
            if not any(pattern in normalized for pattern in suspicious_patterns):
                return True

        return False

    except Exception as e:
        logger.error(f"Error validating ingredient '{ingredient}': {e}")
        # Be permissive on errors
        return True


def validate_ingredients_list(ingredients_str: str) -> tuple[List[str], List[str]]:
    """Validate a list of ingredients and return valid and invalid ones"""
    try:
        # Split by comma and clean up
        ingredients = [ing.strip() for ing in ingredients_str.split(',') if ing.strip()]

        valid_ingredients = []
        invalid_ingredients = []

        for ingredient in ingredients:
            if is_valid_ingredient(ingredient):
                valid_ingredients.append(ingredient)
            else:
                invalid_ingredients.append(ingredient)

        return valid_ingredients, invalid_ingredients
    except Exception as e:
        logger.error(f"Error validating ingredients list: {e}")
        return [], []


@app.route('/validate-ingredients', methods=['POST', 'OPTIONS'])
def validate_ingredients():
    """New endpoint to validate ingredients before processing"""
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
        response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
        return response

    try:
        data = request.get_json()
        logger.info(f"Ingredient validation request data: {data}")

        if not data or 'ingredients' not in data:
            return jsonify({
                'success': False,
                'valid_ingredients': [],
                'invalid_ingredients': [],
                'error': 'Missing ingredients field'
            }), 400

        ingredients_str = data['ingredients']
        valid_ingredients, invalid_ingredients = validate_ingredients_list(ingredients_str)

        return jsonify({
            'success': True,
            'valid_ingredients': valid_ingredients,
            'invalid_ingredients': invalid_ingredients,
            'error': None
        })

    except Exception as e:
        logger.error(f"Ingredient validation error: {str(e)}")
        return jsonify({
            'success': False,
            'valid_ingredients': [],
            'invalid_ingredients': [],
            'error': str(e)
        }), 500


def filter_ingredients_for_dietary(ingredients: List[str], dietary_restriction: str) -> List[str]:
    """Filter out ingredients that conflict with dietary restrictions"""
    try:
        if dietary_restriction == "None":
            return ingredients

        restriction = dietary_restriction.lower().replace("-", "")
        if restriction not in DIETARY_CONFLICTS:
            return ingredients

        conflicts = DIETARY_CONFLICTS[restriction]
        filtered = []

        for ingredient in ingredients:
            normalized = ingredient.lower().strip()

            # Check if ingredient conflicts
            is_conflict = False
            for conflict in conflicts:
                if normalized == conflict or conflict in normalized or normalized in conflict:
                    is_conflict = True
                    break

            if not is_conflict:
                filtered.append(ingredient)

        return filtered
    except Exception as e:
        logger.error(f"Error filtering ingredients for dietary restrictions: {e}")
        return ingredients


def find_flavor(state: State) -> State:
    # Validate ingredients first
    valid_ingredients, _ = validate_ingredients_list(state.ingredients)

    if not valid_ingredients:
        error_message = "No valid ingredients provided. Please provide actual food ingredients."
        return State(
            messages=state.messages + [HumanMessage(content=error_message)],
            ingredients=state.ingredients,
            cuisine=state.cuisine,
            time=state.time,
            meal_type=state.meal_type,
            dietary_restrictions=state.dietary_restrictions
        )

    # Use only valid ingredients
    ingredients_str = ", ".join(valid_ingredients)

    # Build dietary restrictions text
    dietary_text = ""
    if state.dietary_restrictions != "None":
        dietary_text = f" The recipe must be {state.dietary_restrictions} - please ensure all ingredients and cooking methods comply with {state.dietary_restrictions} dietary requirements."

    # Create a strict ingredient list
    ingredient_count = len(valid_ingredients)

    prompt = f"""You are an expert chef creating recipes with STRICT ingredient limitations.

CRITICAL RULES - FOLLOW EXACTLY:
1. You must use ONLY these {ingredient_count} ingredients: {ingredients_str}
2. You cannot add ANY other ingredients, not even salt, pepper, oil, or water unless they are in the list above
3. You cannot suggest substitutions or additional ingredients
4. If you cannot create a complete recipe with ONLY these ingredients, you must create the best possible dish using what's available
5. Do not mention any ingredients not in the provided list
6. If any ingredient seems inappropriate, non-food, or anatomical, respond with 'Invalid Ingredients' instead of a recipe.

Create a {state.meal_type} recipe from {state.cuisine} cuisine that can be completed in {state.time} minutes using EXCLUSIVELY the ingredients listed above.{dietary_text}

Recipe Requirements:
- Use ONLY the {ingredient_count} ingredients provided: {ingredients_str}
- No additional seasonings, oils, or liquids unless specifically listed
- Simple cooking methods that work with the available ingredients
- Realistic portions and cooking times
- Clear, step-by-step instructions

IMPORTANT: Start with a simple, descriptive dish name (NO brackets, NO "Ingredients" as title, NO formatting symbols)

Format your response exactly like this:
Simple Descriptive Dish Name

Ingredients:
- [ingredient 1 with quantity]
- [ingredient 2 with quantity]
- [etc.]

Instructions:
1. [Step 1]
2. [Step 2]
3. [etc.]

Cooking Time: {state.time} minutes

Calories (per serving): [estimated calories per serving], serves [number of people]

Remember: Use ONLY the {ingredient_count} ingredients provided. Do not add anything else.
"""

    try:
        response = llm.invoke(prompt)
        return State(
            messages=state.messages + [response],
            ingredients=ingredients_str,  # Use validated ingredients
            cuisine=state.cuisine,
            time=state.time,
            meal_type=state.meal_type,
            dietary_restrictions=state.dietary_restrictions
        )
    except Exception as e:
        error_message = f"Error generating recipe: {str(e)}"
        logger.error(error_message)
        return State(
            messages=state.messages + [HumanMessage(content=error_message)],
            ingredients=state.ingredients,
            cuisine=state.cuisine,
            time=state.time,
            meal_type=state.meal_type,
            dietary_restrictions=state.dietary_restrictions
        )


graph_builder.add_node("findsFlavor", find_flavor)
graph_builder.add_edge(START, "findsFlavor")
graph_builder.add_edge("findsFlavor", END)

graph = graph_builder.compile()


@app.route('/suggest-ingredients', methods=['POST', 'OPTIONS'])
def suggest_ingredients():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
        response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
        return response

    try:
        data = request.get_json()
        logger.info(f"Ingredient suggestion request data: {data}")

        # Validate input
        if not data or not all(k in data for k in ('base_ingredients', 'cuisine', 'meal_type')):
            error_msg = 'Missing required fields: base_ingredients, cuisine, meal_type'
            return jsonify({
                'success': False,
                'suggested_ingredients': [],
                'error': error_msg
            }), 400

        base_ingredients = data['base_ingredients']
        cuisine = data['cuisine']
        meal_type = data['meal_type']
        dietary_restrictions = data.get('dietary_restrictions', 'None')

        # Validate base ingredients first
        valid_base_ingredients, _ = validate_ingredients_list(base_ingredients)

        if not valid_base_ingredients:
            return jsonify({
                'success': False,
                'suggested_ingredients': [],
                'error': 'No valid base ingredients provided'
            }), 400

        # Parse base ingredients to avoid duplicates
        base_ingredient_list = [ing.strip().lower() for ing in valid_base_ingredients]

        # Build dietary restrictions text
        dietary_text = ""
        if dietary_restrictions != "None":
            dietary_text = f" All suggestions must be compatible with {dietary_restrictions} dietary requirements."

        prompt = f"""You are a culinary expert. Given these base ingredients: {", ".join(valid_base_ingredients)}

        Suggest 15-20 additional ingredients that would work perfectly to create an authentic {meal_type} dish in {cuisine} cuisine.{dietary_text}

        Base ingredients already provided: {", ".join(valid_base_ingredients)}

        Focus on ingredients that are:
        1. Authentic and traditional for {cuisine} cuisine
        2. Appropriate for {meal_type} (breakfast/lunch/dinner/snack)
        3. Complement the base ingredients provided
        4. Commonly available in most kitchens or grocery stores
        5. Include proteins, vegetables, seasonings, and cooking essentials specific to {cuisine} cuisine

        For {cuisine} cuisine and {meal_type}, suggest ingredients that would create an authentic, delicious dish.

        Return ONLY a JSON array of ingredient names, like this:
        ["ingredient1", "ingredient2", "ingredient3", ...]

        Do not include any text before or after the JSON array.
        Do not include any of these base ingredients: {", ".join(valid_base_ingredients)}
        """

        try:
            response = llm.invoke(prompt)
            content = response.content.strip()

            # Clean up the response to extract JSON
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()

            # Parse the JSON response
            suggested_ingredients = json.loads(content)

            # Ensure it's a list of strings
            if not isinstance(suggested_ingredients, list):
                raise ValueError("Response is not a list")

            # Filter out any ingredients that are already in base_ingredients
            filtered_suggestions = [
                ing for ing in suggested_ingredients
                if isinstance(ing, str) and ing.lower() not in base_ingredient_list
            ]

            # Validate suggested ingredients
            final_suggestions = []
            for suggestion in filtered_suggestions:
                if is_valid_ingredient(suggestion):
                    final_suggestions.append(suggestion)

            # Filter based on dietary restrictions
            final_suggestions = filter_ingredients_for_dietary(final_suggestions, dietary_restrictions)

            return jsonify({
                'success': True,
                'suggested_ingredients': final_suggestions[:20],  # Limit to 20
                'error': None
            })

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            logger.error(f"Response content: {content}")
            # Fallback with cuisine-specific ingredients
            fallback_ingredients = get_cuisine_fallback(cuisine, meal_type, base_ingredient_list)
            # Filter fallback ingredients based on dietary restrictions
            fallback_ingredients = filter_ingredients_for_dietary(fallback_ingredients, dietary_restrictions)
            return jsonify({
                'success': True,
                'suggested_ingredients': fallback_ingredients,
                'error': None
            })

    except Exception as e:
        logger.error(f"Ingredient suggestion error: {str(e)}")
        return jsonify({
            'success': False,
            'suggested_ingredients': [],
            'error': str(e)
        }), 500


def get_cuisine_fallback(cuisine, meal_type, base_ingredients):
    """Get cuisine-specific fallback ingredients"""
    cuisine_ingredients = {
        'Italian': ['basil', 'oregano', 'mozzarella', 'parmesan', 'tomato sauce', 'bell peppers',
                    'balsamic vinegar', 'pine nuts', 'sun-dried tomatoes', 'pancetta', 'ricotta'],
        'French': ['thyme', 'rosemary', 'butter', 'cream', 'white wine', 'shallots',
                   'mushrooms', 'gruyere', 'brie', 'dijon mustard', 'tarragon'],
        'Mexican': ['cumin', 'chili powder', 'lime', 'cilantro', 'jalapeños', 'avocado',
                    'corn', 'black beans', 'cotija cheese', 'chipotle', 'poblano peppers'],
        'Asian': ['soy sauce', 'ginger', 'sesame oil', 'green onions', 'rice vinegar', 'chili sauce',
                  'bok choy', 'shiitake mushrooms', 'rice wine', 'hoisin sauce'],
        'Chinese': ['soy sauce', 'ginger', 'sesame oil', 'green onions', 'rice vinegar', 'chili sauce',
                    'bok choy', 'shiitake mushrooms', 'rice wine', 'hoisin sauce'],
        'Indian': ['garam masala', 'turmeric', 'cumin', 'coriander', 'curry powder', 'yogurt',
                   'cardamom', 'cinnamon', 'cloves', 'fenugreek', 'mustard seeds'],
        'Mediterranean': ['feta cheese', 'olives', 'olive oil', 'oregano', 'lemon', 'cucumber',
                          'mint', 'dill', 'pomegranate', 'tahini', 'sumac'],
        'Thai': ['fish sauce', 'lime', 'basil', 'coconut milk', 'lemongrass', 'galangal',
                 'kaffir lime leaves', 'thai chilies', 'palm sugar', 'tamarind paste'],
        'Japanese': ['miso', 'sake', 'mirin', 'nori', 'wasabi', 'pickled ginger', 'edamame',
                     'shiitake', 'tofu', 'dashi', 'panko', 'yuzu']
    }

    # Get cuisine-specific ingredients, default to common ingredients
    ingredients = cuisine_ingredients.get(cuisine, ['salt', 'pepper', 'olive oil', 'garlic', 'onion'])

    # Add common essentials
    essentials = ['salt', 'black pepper', 'olive oil', 'garlic', 'onion', 'butter']
    all_ingredients = list(set(ingredients + essentials))

    # Filter out base ingredients
    filtered = [ing for ing in all_ingredients if ing.lower() not in base_ingredients]

    # Shuffle and return
    import random
    random.shuffle(filtered)
    return filtered[:15]


@app.route('/validate-recipe', methods=['POST', 'OPTIONS'])
def validate_recipe():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
        response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
        return response

    try:
        data = request.get_json()
        logger.info(f"Recipe validation request data: {data}")

        # Validate input
        required_fields = ['available_ingredients', 'cuisine', 'time', 'meal_type']
        if not data or not all(k in data for k in required_fields):
            error_msg = f'Missing required fields: {", ".join(required_fields)}'
            return jsonify({
                'success': False,
                'can_make_recipe': False,
                'error': error_msg
            }), 400

        available_ingredients = data['available_ingredients']
        cuisine = data['cuisine']
        time = int(data['time'])
        meal_type = data['meal_type']
        dietary_restrictions = data.get('dietary_restrictions', 'None')

        # Validate ingredients
        valid_ingredients = []
        for ingredient in available_ingredients:
            if is_valid_ingredient(ingredient):
                valid_ingredients.append(ingredient)

        if not valid_ingredients or len(valid_ingredients) < 3:
            return jsonify({
                'success': True,
                'can_make_recipe': False,
                'missing_ingredients': ["Need at least 3 valid ingredients to make a recipe"],
                'error': None
            })

        # Auto-approve if we have 3 or more valid ingredients
        if len(valid_ingredients) >= 3:
            # Generate a recipe since we're confident this will work
            try:
                dietary_text = ""
                if dietary_restrictions != "None":
                    dietary_text = f" The recipe must be {dietary_restrictions} - please ensure all ingredients and cooking methods comply with {dietary_restrictions} dietary requirements."

                ingredients_str = ", ".join(valid_ingredients)
                ingredient_count = len(valid_ingredients)

                recipe_prompt = f"""You are an expert chef creating recipes with STRICT ingredient limitations.

CRITICAL RULES - FOLLOW EXACTLY:
1. You must use ONLY these {ingredient_count} ingredients: {ingredients_str}
2. You cannot add ANY other ingredients, not even salt, pepper, oil, or water unless they are in the list above
3. You cannot suggest substitutions or additional ingredients
4. If you cannot create a complete recipe with ONLY these ingredients, you must create the best possible dish using what's available
5. If any ingredient seems inappropriate, non-food, or anatomical, respond with 'Invalid Ingredients' instead of a recipe.


Create a {meal_type} recipe from {cuisine} cuisine that can be completed in {time} minutes using EXCLUSIVELY the ingredients listed above.{dietary_text}

IMPORTANT: Start with a simple, descriptive dish name (NO brackets, NO "Ingredients" as title, NO formatting symbols)

Format your response exactly like this:
Simple Descriptive Dish Name

Ingredients:
- [ingredient 1 with quantity]
- [ingredient 2 with quantity]
- [etc.]

Instructions:
1. [Step 1]
2. [Step 2]
3. [etc.]

Cooking Time: {time} minutes

Calories (per serving): [estimated calories per serving], serves [number of people]

Remember: Use ONLY the {ingredient_count} ingredients provided: {ingredients_str}
"""

                response = llm.invoke(recipe_prompt)
                recipe_content = response.content.replace('**', '').replace('*', '').replace('#', '')
                # Remove any brackets from the title
                recipe_content = recipe_content.replace('[', '').replace(']', '')

                return jsonify({
                    'success': True,
                    'can_make_recipe': True,
                    'recipe': recipe_content,
                    'missing_ingredients': [],
                    'error': None
                })
            except Exception as e:
                logger.error(f"Error generating auto-approved recipe: {str(e)}")
                # Fall through to normal validation if auto-generation fails

        # Fallback validation (should rarely be reached now)
        return jsonify({
            'success': True,
            'can_make_recipe': False,
            'recipe': None,
            'missing_ingredients': ["Unable to create recipe with current ingredients"],
            'error': None
        })

    except Exception as e:
        logger.error(f"Recipe validation error: {str(e)}")
        return jsonify({
            'success': False,
            'can_make_recipe': False,
            'error': str(e)
        }), 500


@app.route('/generate-recipe', methods=['POST', 'OPTIONS'])
def generate_recipe():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
        response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
        return response

    try:
        logger.info(f"Received request: {request.method}")
        logger.info(f"Content-Type: {request.content_type}")

        data = request.get_json()
        logger.info(f"Request data: {data}")

        # Validate input
        if not data or not all(k in data for k in ('ingredients', 'cuisine', 'time', 'meal_type')):
            error_msg = 'Missing required fields: ingredients, cuisine, time, meal_type'
            logger.warning(f"Validation error: {error_msg}")
            return jsonify({
                'success': False,
                'recipe': '',
                'error': error_msg
            }), 400

        ingredients = data['ingredients']
        cuisine = data['cuisine']
        time = int(data['time'])
        meal_type = data['meal_type']
        dietary_restrictions = data.get('dietary_restrictions', 'None')

        # Validate ingredients
        valid_ingredients, invalid_ingredients = validate_ingredients_list(ingredients)

        if not valid_ingredients:
            return jsonify({
                'success': False,
                'recipe': '',
                'error': 'No valid ingredients provided. Please provide actual food ingredients.'
            }), 400

        # Use only valid ingredients
        ingredients_str = ", ".join(valid_ingredients)

        user_message = HumanMessage(
            content=f"Ingredients: {ingredients_str}, Cuisine: {cuisine}, Time: {time} minutes, Meal Type: {meal_type}, Dietary: {dietary_restrictions}")

        initial_state = State(
            messages=[user_message],
            ingredients=ingredients_str,
            cuisine=cuisine,
            time=time,
            meal_type=meal_type,
            dietary_restrictions=dietary_restrictions
        )

        result = graph.invoke(initial_state)

        last_message = result["messages"][-1]
        if hasattr(last_message, 'content'):
            recipe_content = last_message.content
        else:
            recipe_content = str(last_message)

        # Clean up the recipe content - remove any markdown or special formatting
        cleaned_recipe = recipe_content.replace('**', '').replace('*', '').replace('#', '').replace('##', '').replace(
            '###', '')
        # Remove any brackets from the title
        cleaned_recipe = cleaned_recipe.replace('[', '').replace(']', '')

        return jsonify({
            'success': True,
            'recipe': cleaned_recipe,
            'error': None
        })

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        return jsonify({
            'success': False,
            'recipe': '',
            'error': str(e)
        }), 500


@app.route('/generate-variations', methods=['POST', 'OPTIONS'])
def generate_variations():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
        response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
        return response

    try:
        logger.info(f"Received variations request: {request.method}")
        data = request.get_json()
        logger.info(f"Variations request data: {data}")

        # Validate input
        if not data or not all(k in data for k in ('ingredients', 'cuisine', 'time', 'meal_type')):
            error_msg = 'Missing required fields: ingredients, cuisine, time, meal_type'
            return jsonify({
                'success': False,
                'variations': [],
                'error': error_msg
            }), 400

        ingredients = data['ingredients']
        cuisine = data['cuisine']
        time = int(data['time'])
        meal_type = data['meal_type']
        dietary_restrictions = data.get('dietary_restrictions', 'None')

        # Validate ingredients
        valid_ingredients, _ = validate_ingredients_list(ingredients)

        if not valid_ingredients:
            return jsonify({
                'success': False,
                'variations': [],
                'error': 'No valid ingredients provided. Please provide actual food ingredients.'
            }), 400

        # Use only valid ingredients
        ingredients_str = ", ".join(valid_ingredients)

        # Generate 3 different recipe variations
        variations = []
        ingredient_count = len(valid_ingredients)

        for i in range(3):
            # Build dietary restrictions text for variations
            dietary_text = ""
            if dietary_restrictions != "None":
                dietary_text = f" The recipe must be {dietary_restrictions} - please ensure all ingredients and cooking methods comply with {dietary_restrictions} dietary requirements."

            variation_prompt = f"""You are an expert chef creating recipe variation #{i + 1} with STRICT ingredient limitations.

CRITICAL RULES - FOLLOW EXACTLY:
1. You must use ONLY these {ingredient_count} ingredients: {ingredients_str}
2. You cannot add ANY other ingredients, not even salt, pepper, oil, or water unless they are in the list above
3. You cannot suggest substitutions or additional ingredients
4. Create a DIFFERENT cooking method or style from the other variations
5. This is variation #{i + 1} - be creative with cooking techniques while using the same ingredients

Create a {meal_type} recipe from {cuisine} cuisine that can be completed in {time} minutes using EXCLUSIVELY the ingredients listed above.{dietary_text}

Cooking style for this variation: {"Grilled/Roasted" if i == 0 else "Sautéed/Pan-fried" if i == 1 else "Steamed/Boiled"}

Format your response EXACTLY like this:
Creative Recipe Name (a descriptive dish name, NOT generic terms, NO brackets)

Ingredients:
- [ingredient 1 with quantity]
- [ingredient 2 with quantity]
- [etc.]

Instructions:
1. [Step 1]
2. [Step 2]
3. [etc.]

Cooking Time: {time} minutes

Calories (per serving): [estimated calories per serving], serves [number of people]

IMPORTANT: Start with a creative, descriptive recipe name that describes the dish, NOT "Ingredients" or generic terms, NO brackets.
Remember: Use ONLY the {ingredient_count} ingredients provided: {ingredients_str}
"""

            try:
                response = llm.invoke(variation_prompt)
                if hasattr(response, 'content'):
                    # Clean up the recipe content
                    cleaned_variation = response.content.replace('**', '').replace('*', '').replace('#', '').replace(
                        '##', '').replace('###', '')
                    # Remove any brackets from the title
                    cleaned_variation = cleaned_variation.replace('[', '').replace(']', '')
                    variations.append(cleaned_variation)
                else:
                    variations.append(str(response))
            except Exception as e:
                logger.error(f"Error generating variation {i + 1}: {str(e)}")
                variations.append(f"Error generating variation {i + 1}: {str(e)}")

        return jsonify({
            'success': True,
            'variations': variations,
            'error': None
        })

    except Exception as e:
        logger.error(f"Variations error occurred: {str(e)}")
        return jsonify({
            'success': False,
            'variations': [],
            'error': str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    logger.info("Starting Foodie Recipe Server...")
    logger.info("Server will be available at: http://localhost:8000")
    logger.info("Health check endpoint: http://localhost:8000/health")
    logger.info("Recipe generation endpoint: http://localhost:8000/generate-recipe")
    logger.info("Recipe variations endpoint: http://localhost:8000/generate-variations")
    logger.info("Ingredient suggestions endpoint: http://localhost:8000/suggest-ingredients")
    logger.info("Recipe validation endpoint: http://localhost:8000/validate-recipe")
    logger.info("Ingredient validation endpoint: http://localhost:8000/validate-ingredients")

    # Production configuration
    app.run(debug=False, host='0.0.0.0', port=8000)