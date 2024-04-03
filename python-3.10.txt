import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
data = pd.read_csv(R"C:\Users\priya\Downloads\DataNeuron_DataScience_Task1\DataNeuron_DataScience_Task1\DataNeuron_Text_Similarity.csv")

# Preprocessing
# For simplicity, assume data preprocessing steps are already done

# Feature Engineering

text_concatenated = data['text1'] + ' ' + data['text2']
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(text_concatenated)
#COLUMN NAME
print(data.columns)
# Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X_text)
#cosine similarity
cos_sim = cosine_similarity(X_tfidf)
# Assign similarity scores to 'similarity_scores' column in DataFrame 'data'
data['similarity_scores'] = cos_sim.diagonal()
# Print DataFrame with similarity scores
print(data)
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, data['similarity_scores'], test_size=0.2, random_state=42)
X_text = data['text1'] + ' ' + data['text2']  # Combine text1 and text2 for each pair
y = data['similarity_scores']

# Model Building
model = LinearRegression()
model.fit(X_train, y_train)
y_train = y_train.astype(float)
# Model Evaluation
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print("Train Score:", train_score)
print("Test Score:", test_score)

# Save the model
joblib.dump(model, 'semantic_similarity_model.pkl')
############################################################
#Part B: Deploying the Model as an API using Flask

from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('text_similarity_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get text inputs from request body
    text1 = request.json['text1']
    text2 = request.json['text2']
    
    # Vectorize the input text
    text_concatenated = text1 + ' ' + text2
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([text_concatenated])
    
    # Predict similarity score
    similarity_score = model.predict(tfidf_matrix)[0]
    
    # Return the result
    return jsonify({'similarity_score': similarity_score})

if __name__ == '__main__':
    app.run(debug=True)
