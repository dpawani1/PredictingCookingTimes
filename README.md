<h1 align="center">Investigation into Predicting Cooking Time using Machine Learning</h1>

<p align="center">By Darsh Pawani (darshvpawani23@gmail.com)</p>

## Overview	

This Data Science project focuses on investigating the factors influencing cooking times and predicting recipe preparation times using machine learning techniques. By analyzing attributes such as the number of ingredients, nutritional proportions, and user ratings, the project aims to provide actionable insights for meal planning and recipe platform optimization. This analysis is conducted on datasets sourced from [food.com](https://www.food.com/), a popular recipe-sharing platform.

----

## Introduction

I chose this project to explore the dynamics of recipe preparation and develop a predictive model for cooking times. By understanding the relationship between key recipe features such as complexity, nutritional proportions, and user feedback, the project aims to provide meaningful insights into cooking trends. The data comes from [food.com](https://www.food.com/) and includes recipe details and user ratings, originally compiled for the research paper *Generating Personalized Recipes from Historical User Preferences* by Majumder et al.

The first dataset, `recipes`, contains 83,782 unique recipes, with each row representing one recipe and its associated details across 14 columns:

| Column             | Description                                                                                     |
|--------------------|-------------------------------------------------------------------------------------------------|
| `name`            | The name of the recipe                                                                          |
| `id`              | A unique identifier for the recipe                                                             |
| `minutes`         | The total time (in minutes) needed to prepare the recipe                                        |
| `contributor_id`  | The ID of the user who submitted the recipe                                                     |
| `submitted`       | The date the recipe was added to the dataset                                                    |
| `tags`            | Categories and tags associated with the recipe                                                 |
| `nutrition`       | Nutrition details including calories, fat, sugar, protein, etc., as percentages of daily values |
| `n_steps`         | Number of steps needed to complete the recipe                                                  |
| `steps`           | The step-by-step instructions for preparing the recipe                                         |
| `description`     | A brief user-provided description of the recipe                                                |
| `ingredients`     | A list of ingredients used in the recipe                                                       |
| `n_ingredients`   | The total number of ingredients used                                                           |

The second dataset, `reviews`, consists of 731,927 rows, each representing a user's interaction with a recipe. It contains five columns:

| Column      | Description                          |
|-------------|--------------------------------------|
| `user_id`   | The ID of the user who left the review|
| `recipe_id` | The ID of the recipe being reviewed  |
| `date`      | The date of the review               |
| `rating`    | The score (out of 5) given to the recipe |
| `review`    | The review text provided by the user |

To prepare the datasets for analysis, we extracted detailed information from the `nutrition` column by splitting it into separate fields, such as `calories`, `protein`, and `sugar`. Additionally, we engineered new features, including `prop_protein`, which represents the proportion of protein calories relative to the total calories. These adjustments allow for a deeper exploration of how recipe attributes like nutritional content and complexity relate to cooking times and user ratings.

This analysis aims to uncover meaningful patterns in recipe preparation, helping contributors optimize recipes to align with user preferences while providing valuable insights for Food.com users.


----

## Data Cleaning and Exploratory Data Analysis

To make the data usable for analysis, I performed cleaning steps such as removing duplicates, handling missing values, and transforming features into usable formats. Key insights from the exploratory analysis include distributions of cooking times, the impact of the number of ingredients, and the proportion of protein in recipes.

To ensure our dataset was ready for analysis and machine learning, I applied several data cleaning steps:

### Data Cleaning Steps

1. **Merge Datasets**:  
   I performed a left merge of the `recipes` and `reviews` datasets using `id` and `recipe_id` as keys. This combined the unique recipes with their corresponding ratings and reviews, ensuring we could analyze both recipe attributes and user interactions.

2. **Check Data Types**:  
   I inspected the data types of all columns to determine appropriate cleaning steps and identified necessary type conversions.

| Column             | Data Type      |
|--------------------|----------------|
| `name`            | object         |
| `id`              | int64          |
| `minutes`         | int64          |
| `contributor_id`  | int64          |
| `submitted`       | object         |
| `tags`            | object         |
| `nutrition`       | object         |
| `n_steps`         | int64          |
| `steps`           | object         |
| `description`     | object         |
| `ingredients`     | object         |
| `n_ingredients`   | int64          |
| `user_id`         | float64        |
| `recipe_id`       | float64        |
| `date`            | object         |
| `rating`          | float64        |
| `review`          | object         |

3. **Handle Missing Ratings**:  
   Ratings of `0` were replaced with `NaN` since ratings are typically on a scale of 1 to 5, and a value of `0` indicates missing data. This step ensured unbiased analysis of recipe ratings.

4. **Add Average Ratings**:  
   A new column, `average_rating`, was created by averaging all user ratings for each recipe. This feature provides an overall measure of user satisfaction with a recipe.

5. **Split Nutrition Column**:  
   The `nutrition` column contained values stored as strings resembling lists (e.g., `[calories, fat, sugar, protein, etc.]`). I split these into separate numerical fields for easier analysis:
   - `calories (#)`
   - `total fat (PDV)`
   - `sugar (PDV)`
   - `protein (PDV)`
   - `sodium (PDV)`
   - `carbohydrates (PDV)`

6. **Convert Dates**:  
   The `submitted` and `date` columns were converted from objects to datetime format. This enabled us to analyze trends and patterns over time if needed.

7. **Add Cooking Complexity Features**:  
   I added interaction terms such as `prop_protein` (proportion of protein calories relative to total calories) and `n_ingredients` multiplied by `prop_protein`. These features provide deeper insights into how recipe complexity impacts cooking time.

---

### Final DataFrame Summary

After cleaning, the dataset contains **234,429 rows** and **27 columns**. Below is a summary of the cleaned columns:

| Column                   | Description                                 | Data Type      |
|--------------------------|---------------------------------------------|----------------|
| `name`                  | Recipe name                                | object         |
| `id`                    | Unique recipe ID                           | int64          |
| `minutes`               | Total preparation time (in minutes)        | int64          |
| `contributor_id`        | User ID who submitted the recipe           | int64          |
| `submitted`             | Date the recipe was submitted              | datetime64[ns] |
| `tags`                  | Tags describing the recipe                 | object         |
| `nutrition`             | Nutrition information (original column)    | object         |
| `n_steps`               | Number of steps in the recipe              | int64          |
| `steps`                 | Step-by-step instructions                  | object         |
| `description`           | Recipe description                         | object         |
| `ingredients`           | List of ingredients                        | object         |
| `n_ingredients`         | Number of ingredients                      | int64          |
| `user_id`               | ID of the user who reviewed the recipe     | float64        |
| `recipe_id`             | ID of the reviewed recipe                  | float64        |
| `date`                  | Date of the review                         | datetime64[ns] |
| `rating`                | Rating given to the recipe (1-5 scale)     | float64        |
| `review`                | Text of the review                         | object         |
| `average_rating`        | Average rating for the recipe              | float64        |
| `calories (#)`          | Total calories in the recipe               | float64        |
| `total fat (PDV)`       | Percent daily value of fat                 | float64        |
| `sugar (PDV)`           | Percent daily value of sugar               | float64        |
| `sodium (PDV)`          | Percent daily value of sodium              | float64        |
| `protein (PDV)`         | Percent daily value of protein             | float64        |
| `saturated fat (PDV)`   | Percent daily value of saturated fat       | float64        |
| `carbohydrates (PDV)`   | Percent daily value of carbohydrates       | float64        |
| `prop_protein`          | Proportion of protein calories             | float64        |

---

### First 5 Rows of Cleaned DataFrame


| name                                | id      | minutes | contributor_id | submitted   | tags                            | n_steps | average_rating | calories | protein_PDV | prop_protein |
|-------------------------------------|---------|---------|----------------|-------------|---------------------------------|---------|----------------|----------|-------------|--------------|
| 1 brownies in the world best ever   | 333281  | 40      | 985201         | 2008-10-27  | ['60-minutes-or-less', ... ]    | 10      | 4.0            | 138.4    | 3.0         | 0.04         |
| 1 in canada chocolate chip cookies  | 453467  | 45      | 1848091        | 2011-04-11  | ['60-minutes-or-less', ... ]    | 12      | 5.0            | 595.1    | 13.0        | 0.04         |
| 412 broccoli casserole              | 306168  | 40      | 50969          | 2008-05-30  | ['60-minutes-or-less', ... ]    | 6       | 5.0            | 194.8    | 22.0        | 0.23         |
| millionaire pound cake              | 286009  | 120     | 461724         | 2008-02-12  | ['time-to-make', ... ]          | 7       | 5.0            | 878.3    | 20.0        | 0.05         |
| 2000 meatloaf                       | 475785  | 90      | 2202916        | 2012-03-06  | ['time-to-make', ... ]          | 17      | 5.0            | 267.0    | 29.0        | 0.22         |

---

## Univariate Analysis

### 1. Histogram of Average Ratings

This histogram shows the distribution of average ratings for recipes. Most recipes are rated positively, with a cluster around 4.0 to 5.0.

<iframe
  src="assets/average_rating_histogram.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

---

### 2. Histogram of Calories (Filtered to 0–5,000)

This histogram visualizes the calorie content of recipes, highlighting that most recipes are relatively low in calories.

<iframe
  src="assets/calories_histogram.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

---

## Bivariate Analysis

### 1. Scatter Plot: Calories vs. Average Rating

This scatter plot examines the relationship between calorie content and average ratings, showing no significant trend.

<iframe
  src="assets/calories_vs_rating.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

---

### 2. Line Plot: Number of Ingredients vs. Average Rating

This line plot shows how the average rating changes with the number of ingredients, revealing potential trends in recipe complexity.

<iframe
  src="assets/ingredients_vs_rating.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

---

## Interesting Aggregates

Below is the pivot table summarizing the mean, median, minimum, and maximum proportions of protein relative to total calories (`prop_protein`) grouped by cooking time (`minutes`):

| Minutes | Mean Protein Proportion | Median Protein Proportion | Min Protein Proportion | Max Protein Proportion |
|---------|--------------------------|---------------------------|------------------------|------------------------|
| 0       | 0.36                    | 0.36                     | 0.36                  | 0.36                  |
| 1       | 0.05                    | 0.00                     | 0.00                  | 0.47                  |
| 2       | 0.06                    | 0.02                     | 0.00                  | 0.68                  |
| 3       | 0.06                    | 0.03                     | 0.00                  | 0.45                  |
| 4       | 0.11                    | 0.08                     | 0.00                  | 0.73                  |
| ...     | ...                     | ...                      | ...                   | ...                   |
| 127     | 0.15                    | 0.17                     | 0.00                  | 0.29                  |
| 128     | 0.12                    | 0.06                     | 0.01                  | 0.49                  |
| 129     | 0.22                    | 0.24                     | 0.10                  | 0.32                  |
| 130     | 0.16                    | 0.12                     | 0.00                  | 0.67                  |
| 132     | 0.13                    | 0.10                     | 0.03                  | 0.55                  |

This pivot table highlights how the proportion of protein changes with cooking time. For shorter cooking times, protein proportions are more varied, while longer cooking times show more consistency, likely due to the nature of high-protein recipes like stews or roasts.



----

## Assessment of Missingness

### NMAR Analysis

I believe that the missingness of the `average_rating` column is **Not Missing At Random (NMAR)**. This is because users are less likely to leave a rating if they feel indifferent about the recipe or if they did not cook it. For example, users who felt strongly about a recipe, whether positively or negatively, would be more motivated to leave a rating. Indifferent users, on the other hand, might not find it worth their time to interact further by leaving a rating.

Additionally, users who did not actually attempt the recipe might avoid rating it altogether, as they might not feel qualified to provide feedback. To better understand this missingness and determine if it could instead be Missing At Random (MAR), I would need additional data, such as user interaction logs indicating whether a recipe was viewed or saved without being rated.

## Assessment of Missingness

### Proportion of Protein and Average Rating Missingness

I tested whether the missingness in `average_rating` depends on the proportion of protein in a recipe (`prop_protein`).

- **Null Hypothesis**: The missingness of `average_rating` does not depend on `prop_protein`.
- **Alternate Hypothesis**: The missingness of `average_rating` depends on `prop_protein`.
- **Observed Statistic**: 0.0109
- **P-value**: 0.0  
- **Decision**: Since the p-value is less than the significance level of 0.05, I reject the null hypothesis. This suggests that the missingness of `average_rating` is dependent on `prop_protein`.

<iframe
  src="assets/prop_protein_missingness.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

---

### Cooking Time and Average Rating Missingness

I tested whether the missingness in `average_rating` depends on the cooking time of the recipe (`minutes`).

- **Null Hypothesis**: The missingness of `average_rating` does not depend on `minutes`.
- **Alternate Hypothesis**: The missingness of `average_rating` depends on `minutes`.
- **Observed Statistic**: 117.34
- **P-value**: 0.036  
- **Decision**: Since the p-value is less than the significance level of 0.05, I reject the null hypothesis. This suggests that the missingness of `average_rating` is dependent on `minutes`.

<iframe
  src="assets/minutes_missingness.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

---

----

## Hypothesis Testing

**Null Hypothesis**:
The mean rating of protein recipes is equal to that of non-protein recipes.

**Alternative Hypothesis**:
The mean rating of protein recipes is different from that of non-protein recipes.

Using a permutation test, I found a statistically significant difference in ratings, suggesting that protein-rich recipes tend to have distinct ratings compared to others.

----

## Framing a Prediction Problem

I want to predict the **cooking time (in minutes)** for recipes based on their attributes, such as the number of ingredients, nutritional information, and tags. This is a regression problem, evaluated using **Mean Absolute Error (MAE)** to ensure interpretability and robustness against outliers.

----

## Baseline Model

The baseline model uses a simple **Linear Regression** trained on `n_ingredients` and `tags`. It provides a starting point with a test MAE of approximately **118.47** minutes, highlighting the need for improved feature engineering and modeling techniques.

----

## Final Model

The final model incorporates additional features, including the proportion of protein (`prop_protein`) and interaction terms. A **Random Forest Regressor** was used to account for non-linear relationships, resulting in a test MAE of approximately **115.71** minutes. This improvement demonstrates the value of feature engineering in enhancing model performance.

----

## Fairness Analysis

To assess fairness, I examined whether the model’s predictions varied significantly for recipes categorized as "protein-rich" (high `prop_protein`) versus others. By comparing prediction errors across these groups, I ensured that the model performs equitably regardless of recipe composition.
