# Product-Recommendation-System

This notebook contains 3 checkpoints that, in conjunction, build a Recommendation System.

**Dependencies**
The notebook requires standard data science libraries, including:
- Python 3.8+
- Pandas for data manipulation
- NumPy for numerical operations
- scikit-learn for normalization and encoding
- Google Colab runtime for environment setup and Drive integration

## Phase 1 - Exploratory Data Analysis
This notebook shows the steps I completed for data visualization, data preprocessing and feature engineering of the given dataset, preparing user, item, and interaction data for later model training and evaluation.

**Core objectives:**
- Connect and load data from Google Drive
- Process raw user, item, and interaction logs
- Engineer behavioral features
- Normalize and encode numerical and categorical features
- Merge all relevant information into a single model-ready dataset

**Workflow Summary**
1. Data Access and Setup
  The notebook runs in Google Colab, with Google Drive mounted to access the datasets.
Data files are stored in a structured directory under the project’s drive folder.
A helper function is used to systematically load and combine files for users, items, and interaction histories into Pandas DataFrames.

3. User Feature Engineering
  User-level features are engineered from the historical interaction log to capture behavioral patterns.
This includes computing the total number of interactions per user and each user’s rate of positive interactions (e.g., clicks or purchases marked as relevant).
These new features are merged with the user profile dataset, filling missing values as needed.
This step quantifies user engagement and preference tendencies, which are key inputs for personalized recommendation models.

3. Item Feature Engineering
  Item-level attributes are enhanced by calculating item popularity based on the number of interactions across users.
This feature reflects how frequently an item is engaged with and serves as a proxy for general appeal or demand.
The popularity metric is integrated into the item dataset to enrich item representations.

4. Feature Normalization
  All continuous user and item features are normalized to ensure they share a common scale and distribution.
Normalization helps prevent bias in downstream models where differing feature magnitudes might otherwise distort learning.
Typical normalized features include user behavior statistics, item prices, and popularity measures.

5. Merging Datasets
  Once all user, item, and interaction features are prepared, they are merged into a single comprehensive dataset.
This unified dataset aligns each interaction record with the corresponding user and item features, ensuring consistency across all dimensions.
A validation step confirms that there are no missing or mismatched values in the merged data.

6. Categorical Feature Encoding
  Categorical attributes, such as user segments or item categories, are transformed into numerical format through one-hot encoding.
This ensures compatibility with machine learning algorithms that require numerical inputs.
Encoding also allows categorical variables to contribute meaningfully to similarity and distance computations in recommendation models.

**Output**
  At the end of this checkpoint, the notebook produces a clean, feature-rich, and model-ready dataset.
All numeric attributes are standardized, categorical features are encoded, and user-item interactions are linked to their corresponding metadata.

**Notes**
- The feature engineering workflow is modular, allowing additional attributes to be added easily in later stages.
- Normalization and encoding steps are reusable for preprocessing new or unseen data.
- Data validation ensures that no missing values persist before moving to model training.
- This checkpoint establishes the foundation for all downstream recommendation system components.

## Checkpoint 2 
