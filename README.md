# Email Spam Detection Machine Learning Project

**Introduction:**

This project focuses on building a machine learning model to classify emails as spam or not spam.

**Data Cleaning and Preprocessing:**

The project starts with importing the necessary libraries and loading the dataset using Pandas. The dataset contains two columns: 'Message' and 'Category' (ham or spam). To ensure data quality, duplicates are removed, and the 'ham' category is relabeled as 'Not Spam' for clarity.

**Data Splitting:**

The dataset is divided into training and testing sets, a common practice in machine learning. The 'Message' column serves as the input dataset, and the 'Category' column is the output dataset. The data is split into 80% for training and 20% for testing using the `train_test_split` function from scikit-learn.

**Text to Numeric Conversion:**

Since machine learning models require numerical input, the project employs the CountVectorizer from scikit-learn to convert text data into a numerical format. This process involves creating a vector that represents the frequency of words in each document (email).

**Model Creation and Training:**

The Multinomial Naive Bayes algorithm is chosen as the classification model. This algorithm is suitable for text classification tasks. The model is trained using the training features (converted text data) and corresponding output labels (spam or not spam).

**Model Evaluation:**

The trained model is evaluated using the testing set, and its accuracy is calculated. The project achieves a high accuracy rate, indicating the model's effectiveness in distinguishing between spam and non-spam emails.

**Real-time Prediction:**

The project includes a function (`predict`) that allows users to input a message, and the model predicts whether it is spam or not. This feature demonstrates the practical application of the machine learning model in real-time scenarios.

**Integration with Streamlit:**

To make the project more user-friendly, Streamlit, a Python library for creating web applications, is integrated. The Streamlit app prompts users to enter an email message, and upon clicking the "Validate" button, the model predicts whether the message is spam or not. The result is displayed to the user, enhancing accessibility and usability.

In conclusion, the "Email Spam Detection" project successfully demonstrates the application of machine learning in identifying and filtering spam emails. The combination of effective data preprocessing, model training, and integration with Streamlit provides a user-friendly and efficient tool for email security.
