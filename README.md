# Product-Recommendation-System

This notebook contains 3 checkpoints that, in conjunction, build a Recommendation System.

Please check out my project [write-up](https://docs.google.com/document/d/17RDXD0kMCzl_XkmpfTTBI2lRsimZZQY_DdZ923ZTN3k/edit?usp=sharing) and [presentation](https://docs.google.com/presentation/d/1-yGsDi_ueHYla5nPrs3N8h1gRC3RBtOotfY9C6r4uKI/edit?usp=sharing) for more detailed information reguarding this project! 


**Dependencies**
The notebook requires standard data science libraries, including:
- Python 3.8+
- Pandas for data manipulation
- NumPy for numerical operations
- scikit-learn for normalization and encoding
- Google Colab runtime for environment setup and Drive integration

## Checkpoint 1/3 - Exploratory Data Analysis
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

## Checkpoint 2/3 - Sequential Modeling
This checkpoint extends the preprocessed data from Checkpoint 1 into a sequence modeling framework for user behavior prediction. The primary objective is to transform static interaction data into ordered purchase sequences suitable for recurrent neural network (RNN)–based recommendation systems.

**Data Transformation**
- User histories are converted into sequential representations where each row corresponds to a user’s ordered list of interacted items.
- Because users vary in activity, sequences are padded with placeholder values (e.g., −1) to achieve uniform input lengths across the dataset.
- The resulting structured data is saved as updated_chckpt2_data.csv, forming the input to sequence-based models.

**Model Overview**
A Recurrent Neural Network (RNN) architecture is introduced to capture temporal dependencies in user–item interactions.
Model configuration:
- Embedding dimension: 64
- Hidden units: 256
- Layers: 2
- Dropout: 0.3
- Learning rate: 0.001
This structure enables the model to learn user intent over time, improving recommendation accuracy for sequential purchase or interaction prediction.

The checkpoint includes integration with the Sim4Rec framework, providing utilities for dataset management, training loops, and evaluation of sequential recommendation models.

**Outcome**
By the end of Checkpoint 2, the project transitions from static feature-based preprocessing to dynamic, order-aware modeling, setting the stage for later experiments with RNNs and other sequence-aware architectures (e.g., GRU, LSTM, or Transformer-based recommenders).

## Checkpoint 3/3 - Final Integration and Feature Consolidation
This final checkpoint refines and consolidates all previous preprocessing and modeling work, producing a fully cleaned, standardized, and integrated dataset suitable for advanced recommendation architectures.

**Data Reconstruction**
- Data is reloaded from the previous checkpoint outputs, ensuring consistency across user, item, and interaction components.
- User and item features are recomputed to confirm integrity, including total interactions, positive interaction rate, and item popularity.
- These are merged into unified data structures for further processing and experimentation.

**Feature Refinement**
- Continuous variables (such as user engagement statistics and item popularity) are re-normalized to maintain feature stability and prevent scale imbalance.
- Categorical variables are re-encoded through one-hot encoding to preserve consistency between checkpoints.
- Final verification ensures no missing or misaligned data entries remain.

**Model Readiness**
Although not fully executed in this checkpoint, the inclusion of optional graph-based modeling tools (such as PyTorch Geometric) suggests potential expansion toward Graph Neural Network (GNN) or hybrid recommendation systems in future work.
This setup allows the dataset to serve as the foundation for both sequence-aware and relationship-aware modeling approaches.

**Outcome**
By the end of Checkpoint 3, the project has achieved a fully operational data pipeline, capable of producing model-ready datasets for diverse recommendation algorithms.
It concludes the preprocessing phase and transitions the project toward model experimentation, evaluation, and deployment stages.
