# Resume Parser AI Project

## Project Scope and Objectives

**Overview**:  
The Resume Parser AI Project aims to develop an AI-driven system for automating the resume screening process. The system will evaluate resumes based on contextual relevance, focusing on extracting key features such as skills and job titles, and providing a scoring mechanism to rank resumes effectively.

**Main Objectives**:
- **Data Analysis**: Process and clean resume data to ensure it is suitable for model training.
- **Feature Extraction**: Identify and extract relevant features like skills and job titles from resumes.
- **Model Development**: Implement and train machine learning models to classify and score resumes.
- **Evaluation**: Assess the performance of the model using various metrics and validate it with a validation dataset (`df_valid`).
- **Deployment**: Prepare the system for practical use, with the potential for future integration with applicant tracking systems.

## Procedure

### 1. Data Preparation

- **Load Dataset**: The dataset is loaded from a CSV file containing columns for job categories and resume text.
- **Preprocessing**: Normalize text, handle missing values, and clean up symbols to ensure uniformity.
- **Train-Test Split**: Divide the data into training and validation sets using an 80-20 split.

    ```python
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from fastai.text.all import *

    # Load and preprocess data
    df = pd.read_csv('data/filtered_data.csv', usecols=[1, 4])

    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)
    ```

### 2. Text Normalization

- **Lowercasing**: Convert all text to lowercase.
- **Remove Punctuation and Special Characters**: Clean the text data to remove any unnecessary symbols.
- **Tokenization and Lemmatization**: Tokenize the text into words and apply lemmatization to reduce words to their base forms.

    ```python
    import re

    # Define text normalization function
    def normalize_text(text):
        # Remove years (4 consecutive digits)
        text_without_years = re.sub(r'\b\d{4}\b', '', text)
        # Remove special characters except for spaces
        pattern = r'[^a-zA-Z0-9\s]'
        clean_text = re.sub(pattern, '', text_without_years)
        # Replace any kind of line breaks with a space
        cleaned_text = re.sub(r'[\r\n]+', ' ', clean_text)
        # Remove extra spaces
        final_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return final_text

    # Apply normalization
    df['Resume'] = df['Resume'].apply(normalize_text)
    ```

### 3. Model Development and Training

- **Clustering**: Implement clustering algorithms to group similar skills and terms.
- **Classification**: Train a logistic regression model for resume classification.

    ```python
    from fastai.text.all import *

    # Define DataLoaders
    dls = TextDataLoaders.from_df(train_df,
                              valid_df=valid_df,
                              text_col='Normalized_Resume',
                              label_col='Category',
                              is_lm=False,
                              bs=32)

    # Create a Learner
    learn = text_classifier_learner(dls, AWD_LSTM, loss_func=CrossEntropyLossFlat(), metrics=[accuracy])

    # Train the model
    learn.fit_one_cycle(8, 0.03)
    ```

### 4. Resume Scoring Mechanism

- **Define Scoring Criteria**: Establish criteria for scoring resumes based on skills, experience, and job title relevance.
- **Develop Scoring Algorithm**: Implement an algorithm to assign scores to resumes from 0 to 10.
- **Integrate Scoring with Clustering**: Combine clustering results with the scoring algorithm to evaluate resumes effectively.

    ```python
    from sklearn.metrics import classification_report, confusion_matrix

    # Validate with df_valid
    # Get predictions and targets
    preds, targets = learn.get_preds(dl=dls.valid)

    # Classification Report
    print("Classification Report:")
    print(classification_report(val_targets, val_preds.argmax(dim=1)))

    # Confusion Matrix
    print("Confusion Matrix:")
    print(confusion_matrix(val_targets, val_preds.argmax(dim=1)))
    ```

## Conclusion

The Resume Parser AI Project demonstrates a robust approach to automating the resume screening process using machine learning techniques. By focusing on data preprocessing, feature extraction, and model development, the project aims to improve the efficiency and accuracy of resume evaluation.

**Future Work**: Future enhancements may include integrating the system with applicant tracking systems, handling more diverse datasets, and refining the model based on stakeholder feedback and real-world performance.

**Contact**: For more information or contributions, please reach out to Peter Datsenko.
