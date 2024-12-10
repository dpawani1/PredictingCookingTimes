<h1 align="center">Investigation into Predicting Cooking Time using Machine Learning</h1>

<p align="center">By Darsh Pawani (darshvpawani23@gmail.com)</p>

## Overview	

This Data Science project focuses on investigating the factors influencing cooking times and predicting recipe preparation times using machine learning techniques. By analyzing attributes such as the number of ingredients, nutritional proportions, and user ratings, the project aims to provide actionable insights for meal planning and recipe platform optimization. This analysis is conducted on datasets sourced from [food.com](https://www.food.com/), a popular recipe-sharing platform.

----

## Introduction

I chose this project to explore the dynamics of recipe preparation and develop a predictive model for cooking times. By understanding the relationship between key recipe features such as complexity, nutritional proportions, and user feedback, the project aims to provide meaningful insights into cooking trends. The data comes from [food.com](https://www.food.com/) and includes recipe details and user ratings, originally compiled for the research paper *Generating Personalized Recipes from Historical User Preferences* by Majumder et al.

----

## Data Cleaning and Exploratory Data Analysis

To make the data usable for analysis, I performed cleaning steps such as removing duplicates, handling missing values, and transforming features into usable formats. Key insights from the exploratory analysis include distributions of cooking times, the impact of the number of ingredients, and the proportion of protein in recipes.

----

## Assessment of Missingness

- **NMAR**: Missing data in some columns, such as `average_rating`, is likely Not Missing At Random (NMAR) because users only rate recipes they interact with. 
- **MAR/MCAR**: Other missing values, like `tags` or `nutrition`, may be Missing at Random (MAR) due to inconsistencies in data entry.

----

## Hypothesis Testing

**Null Hypothesis**:
The mean rating of protein recipes is equal to that of non-protein recipes.

**Alternative Hypothesis**:
The mean rating of protein recipes is different from that of non-protein recipes.

Using a permutation test, we found a statistically significant difference in ratings, suggesting that protein-rich recipes tend to have distinct ratings compared to others.

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
