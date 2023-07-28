# Advanced Content Moderation: Social Media Comment Classifier using LLMs and ML

## Overview:

Content moderation is a critical aspect of managing digital platforms and ensuring a safe and engaging user experience. However, it encounters numerous challenges when confronted with flagged content that is written in slang, dialects, unclear context, and even misspelled words. These forms of communication can be difficult for content moderators to understand, leading to potential misinterpretations and inaccurate labeling of content as violating or non-violating. Such errors not only waste valuable time but also undermine the effectiveness of content moderation efforts.

The prevalence of keywords indicating potential violations further complicates the content moderation process. While these keywords can serve as initial indicators, relying solely on their presence can result in false positives or negatives. Without a comprehensive understanding of the message's intent and the context in which it is used, content moderators are at risk of misjudging the content's true nature. This leads to inconsistencies, delays, and potential frustrations for users seeking prompt resolution.

These challenges extend beyond content moderators to quality analysts who play a vital role in ensuring accurate content assessments. Misinterpretations in the initial moderation phase can trickle down and affect subsequent analysis, impacting the overall efficiency and effectiveness of the content moderation workflow. The accumulation of errors and inefficiencies slows down the process, preventing timely actions on flagged content and jeopardizing the platform's ability to provide a safe environment for users.

## Objective

The main goal of this project is to tackle these issues head-on by providing an efficient and effective solution using Large Language Models (LLMs) and Machine Learning (ML). LLMs leverage natural language processing to understand content written in diverse forms. The seamless integration of LLMs and ML enhances the content moderation process. Moderators equipped with these tools can make informed decisions, reducing misinterpretations and errors. Increased productivity can be achieved as more flagged items are addressed per hour. Swift resolution of flagged content enhances user satisfaction and fosters a safer online environment. This solution also benefits quality analysts who rely on accurate content assessments. By addressing moderation challenges, analysts receive correctly labeled and interpreted content, minimizing errors in subsequent analysis. This streamlines the content moderation and assessment workflow, enhancing overall efficiency and paving the way for a new era of AI-driven techniques where user satisfaction and platform safety are at the forefront.

## Table of Contents

1. [Datasets](#datasets)
2. [Data Preprocessing](#data-preprocessing)
3. [Machine Learning Models](#machine-learning-models)
4. [Flask App](#flask-app)
5. [Instructions](#instructions)

## Datasets <a name="datasets"></a>

Datasets were extracted from Kaggle due to privacy issues when handling social media platforms. The datasets used can be accessed below:

- [Hate Comment Dataset](https://www.kaggle.com/datasets/subhajeetdas/hate-comment)
- [Suicide Watch Dataset](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)
- [Sexually Explicit Comments Dataset](https://www.kaggle.com/datasets/harsh03/sexually-explicit-comments)
- [Cyberbullying Dataset](https://www.kaggle.com/datasets/saurabhshahane/cyberbullying-dataset)
- [Twitter Sentiment Analysis Dataset](https://www.kaggle.com/code/paoloripamonti/twitter-sentiment-analysis)

The datasets were cleaned, tokenized, had stop words removed, stemmed, and lemmatized. Afterward, a technique of undersampling was used to create the best dataset possible for training the model.

## Data Preprocessing <a name="data-preprocessing"></a>

The cleaned and undersampled data was split into training and test sets. Three different machine learning models were trained on the data: linear regression, random forest, and Support Vector Classifier (SVC). The model with the best results, in this case, SVC, was chosen to be applied in the creation of the Flask app.

## Machine Learning Models <a name="machine-learning-models"></a>

The Support Vector Classifier (SVC) model was trained using the preprocessed data. This model showed the best performance in classifying social media comments into different categories based on their content.

## Flask App <a name="flask-app"></a>

The visual application of the project can be experienced in the Flask app. The Flask app allows users to enter comments, and the trained SVC model will classify them into different violation categories. Below are the relevant files for the Flask app:

### main.py <a name="main.py"></a>

```python
# 
from flask import Flask, render_template, request, jsonify
import pickle
import nltk
from nltk.corpus import stopwords
import spacy

app = Flask(__name__)

# Load your model from the pickle file
with open('svm_model_final.pkl', 'rb') as file:
    svc_model = pickle.load(file)

# Load the spaCy NLP model (adjust for your language, e.g., 'en_core_web_lg' for English)
nlp = spacy.load('en_core_web_lg')

# Define the stopwords for your language (e.g., English)
stopwords = set(stopwords.words('english'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.json.get('user_input', '')  # Get the comment from the JSON payload
        tokens = [token.text for token in nlp(user_input)]
        tokens = [token for token in tokens if token not in stopwords]
        phrase_vector = nlp(user_input).vector
        phrase_vector = phrase_vector.reshape(1, -1)
        predicted_label = svc_model.predict(phrase_vector)[0]
        
        # Map the predicted label to the corresponding message
        messages = {
            0: "Your comment is under General Guidelines.",
            1: "Your comment cannot be posted because it involves Suicide-related Content.",
            2: "Your comment cannot be posted because it contains Sexually Explicit Content.",
            3: "Your comment cannot be posted because it includes Hateful Language.",
            4: "Your comment cannot be posted because it involves Bullying Behavior."
        }
        predicted_message = messages.get(predicted_label, "Unable to determine the violation category.")
        
        return jsonify({'predicted_message': predicted_message})

    return render_template('site.html', predicted_message=None)

if __name__ == '__main__':
    app.run(debug=True)

```

### site.html <a name="site.html"></a>

```html

<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8" />
    <title>Comment Classifier</title>
    <link href="https://fonts.googleapis.com/css2?family=Copperplate&display=swap" rel="stylesheet">
    <style>
        body {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background-color: #f0f0f0;
            font-family: 'Copperplate', sans-serif;
        }
        h1 {
            margin-top: 25px;
            font-size: 50px;
        }
    </style>
</head>
<body>
    <h1>COMMENT CLASSIFIER</h1>
    <div class="instructions">
        <p><strong>Instructions:</strong>
            <br>1. Hover over the comment area to see a tooltip.
            <br>2. Click on the comment area to submit a comment.
            <br>3. After submitting the comment, the classification result will be shown.
        </p>
    </div>
    <div id="image-container">
        <img src="https://cdn.icon-icons.com/icons2/1283/PNG/512/1497619950-jd09_85153.png" alt="Image" usemap="#image-map">
        <div id="comment-tooltip">Click here to comment</div>
    </div>

    <map name="image-map">
        <area id="comment-area" alt="Comment Area" title="Comment Area" coords="197,216,313,316" shape="rect">
    </map>

    <div class="backdrop"></div>

    <div id="popup-container">
        <form id="comment-form">
            <label for="user_comment">Enter a comment:</label><br>
            <input type="text" id="user_comment" name="user_comment"><br>
            <button type="submit">Submit</button>
        </form>
    </div>

    <div id="classification-popup">
        <p id="classification-result"></p>
    </div>

    <script>
    </script>
</body>
</html>

```

## Instructions <a name="instructions"></a>

To use the Comment Classifier Flask app, follow these instructions:

1. Hover over the comment area to see a tooltip.
2. Click on the comment area to submit a comment.
3. After submitting the comment, the classification result will be shown.

The Flask app will then use the trained SVC model to predict the category of the entered comment. The classification result will be displayed on the web page.

Note: The Flask app uses the spaCy library for natural language processing and nltk for stopwords removal. Make sure to have these libraries installed to run the Flask app successfully.


### Thank you! <a name="thank you"></a>

Feel free to use and contribute to this project to enhance content moderation on digital platforms and promote a safer online environment for all users.
