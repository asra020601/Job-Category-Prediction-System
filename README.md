# Job-Category-Prediction-System
This repository contains the code and documentation for a job category prediction system. This system takes a resume as input and identifies the most relevant job categories and descriptions, leveraging natural language processing (NLP) techniques.

Introduction
The Job Category Prediction System aims to streamline the process of matching resumes to appropriate job categories. Instead of predicting specific job roles, which can be highly granular, this system focuses on broader job categories, providing a more generalized and adaptable solution for resume analysis and job recommendations.

How It Works
The system operates through a series of sequential steps, combining data preprocessing with advanced NLP models.

1. Data Preparation
This initial phase focuses on getting the raw data ready for analysis.

Importing Libraries and Dataset: Essential Python libraries, such as pandas for data manipulation, are imported. The core data3.csv dataset, containing resume information, is loaded.

Column Cleaning: Irrelevant columns, such as Job Applicant Name, are removed from the dataset as they do not contribute to the job matching logic.

Filtering Relevant Data: The dataset is filtered to include only entries where Best Match is 1. This ensures that the system trains and operates on high-quality, relevant resume-job description pairs.

2. Job Category Generation
This section details how the system creates and utilizes job categories.

Defining Job Categories: A comprehensive, predefined list of broad job categories is established. These categories serve as the target labels for our prediction model.

Loading Sentence Transformers: The system utilizes a pre-trained SentenceTransformer model, specifically 'all-MiniLM-L6-v2'. This model is chosen for its balance of performance and efficiency, enabling the conversion of text (resumes and job descriptions) into numerical vector representations (embeddings). The fundamental principle here is to generate job categories from resume descriptions based on semantic similarity.

Embedding Categories: Each of the predefined job categories is transformed into a numerical embedding using the loaded SentenceTransformer model. This allows for mathematical comparison and similarity calculations between categories and resume content.

3. Skill Recognition and Categorization
This is where the core matching logic for individual resumes takes place.

Categorization Function: A dedicated function processes each resume's skills:

It parses a comma-separated string of skills into a structured list.

These individual skills are then encoded into numerical embeddings using the SentenceTransformer.

An average embedding is computed from the individual skill embeddings to represent the overall skill profile of the resume.

Cosine Similarity for Matching: The cosine similarity metric is employed to calculate the similarity between the resume's aggregated skill embedding and the embeddings of all predefined job categories. Cosine similarity measures the angular distance between two vectors, with values closer to 1 indicating higher similarity.

The job category exhibiting the highest cosine similarity score is identified as the most probable match for the given resume.

Synthetic Job Category Generation: Through this process, a "synthetic" job category is assigned to each resume by finding the best match between its skills and the predefined categories.

Applying Categories to Data: The newly predicted job categories are appended as a new column (job_category) to the main dataset, enriching the data for further analysis.

4. Top Job Role Prediction
This phase focuses on identifying the most relevant job roles and descriptions based on user input.

Importing CountVectorizer: The CountVectorizer from sklearn.feature_extraction.text is imported. This tool is used to convert text documents into a matrix of token (word) counts, a common technique for text representation in NLP.

Fitting CountVectorizer: The CountVectorizer is fitted to the Resume column of the filtered dataset. This step builds a vocabulary of all unique words present in the resumes and prepares the vectorizer to transform new text inputs into numerical vectors.

Calculating Cosine Similarity for Resumes: After vectorization, cosine similarity is calculated between all resume vectors. This generates a similarity matrix, where each cell represents the similarity score between any two resumes in the dataset.

Prediction Function: A robust function is developed to handle user queries:

It accepts a user's input (e.g., a new resume, a list of skills, or a job description).

This input is then vectorized using the pre-fitted CountVectorizer.

The similarity score between the user's input vector and all existing resume vectors is computed using the pre-calculated cosine similarity matrix.

The function then identifies the top five unique job categories and their corresponding job descriptions that exhibit the highest similarity scores to the user's input.

The results are sorted in descending order based on their similarity ranking, presenting the most relevant matches first.

Setup and Installation
To set up and run this project locally, follow these steps:

Clone the repository:

git clone ]https://github.com/asra020601/Job-Category-Prediction-System/blob/main/main2.ipynb
cd Job-Category-Prediction-System/blob/main/main2.ipynb

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows: `venv\Scripts\activate`

Install the required libraries:

pip install pandas scikit-learn sentence-transformers torch

Ensure you have data3.csv in your project directory. (This file is assumed to be present for the system to function.)

Usage
To use the system, you can run the main script (e.g., main.py if your code is structured that way) and provide a resume or skill set as input.

Example (assuming your prediction logic is in a function like get_unique_job_categories):

# Example of how to use the prediction function
from your_module import get_unique_job_categories # Adjust 'your_module' as needed

user_resume_input = "Proficient in Market Research, Investment Analysis, Forecasting, Budgeting, Corporate Finance, with senior-level experience in the field. Holds a Bachelors degree. Holds certifications such as Certified Financial Planner (CFP). Skilled in delivering results and adapting to dynamic environments."

top_matches = get_unique_job_categories(user_resume_input)

print("Top 5 Job Categories and Descriptions:")
for category, description in top_matches:
    print(f"- Category: {category}\n  Description: {description}\n")

Testing and Validation
The system includes a testing phase where sample inputs are used to validate the accuracy and relevance of the predicted job categories and descriptions. The output demonstrates the system's ability to provide meaningful recommendations based on the input resume.


