Fake News Detection with Machine Learning

This project focuses on building a machine learning pipeline that can automatically classify news headlines as either factual or misleading. The goal is to use natural language processing (NLP) techniques combined with supervised learning models to detect fake news content efficiently and accurately.

The dataset used contains over 10,000 headlines, each labeled with a binary outcome: 0 for factual or not misleading headlines and 1 for misleading or fake ones. These headlines were cleaned and preprocessed to remove punctuation, convert all text to lowercase, and strip out numeric characters. The cleaned text data was then transformed into numerical features using TF-IDF vectorization, which helps quantify the importance of words relative to the dataset.

The machine learning pipeline evaluates three classification models: Logistic Regression, Random Forest Classifier, and Multinomial Naive Bayes. The dataset was split into an 80/20 training and testing set, and each model was trained using a pipeline that included both TF-IDF transformation and the classifier. To improve model performance, hyperparameter tuning was applied to the Random Forest model using Grid Search with cross-validation.

After training, the models were evaluated based on their accuracy, precision, recall, F1-score, and AUC-ROC. Among the models tested, the Random Forest Classifier performed best, achieving the highest scores across most metrics, particularly in recall and F1-score, making it the most effective model for this binary classification task. The Logistic Regression model performed well in terms of AUC-ROC, while the Naive Bayes model, although high in precision, significantly underperformed in recall and F1-score.

The project was developed in Python and uses libraries such as pandas, scikit-learn, and matplotlib. To run the project, users can simply clone the repository, install the required packages, and execute the main script to see the pipeline in action. The code includes detailed steps for preprocessing, training, evaluation, and visualization of results including confusion matrices and performance charts.

Looking ahead, this project can be expanded in several ways. Incorporating more advanced models like LSTM or BERT could improve accuracy and generalization. Additionally, including metadata such as the source, author, or publication date of each headline might enhance the model’s ability to detect fake news. Finally, deploying the model using a web framework like Streamlit or Flask could make it accessible as an interactive web application.

This project was created as part of a machine learning course and serves as a hands-on example of how text classification techniques can be applied to real-world problems like misinformation detection. If you have any questions, feedback, or collaboration ideas, feel free to reach out to the author.

