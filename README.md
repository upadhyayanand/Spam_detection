**📩 SMS Spam Detection using Machine Learning:**

This project is a Machine Learning–based SMS Spam Detection system that classifies messages as Spam or Ham (Not Spam) using Natural Language Processing (NLP) techniques.

The model converts text into numerical features using Count Vectorization and TF-IDF, then applies multiple ML algorithms to find the best-performing classifier.


----------------------------------------------



**🚀 Features:**

-Text preprocessing and vectorization

-Multiple ML models comparison

-Automatic best model selection based on accuracy

-Trained model saved using joblib

-Ready for real-time SMS prediction


------------------------------------------



**🧠 Machine Learning Models Used:**

-Naive Bayes (MultinomialNB)

-Logistic Regression

-Linear Support Vector Machine (LinearSVC)


-------------------------------------



**🔧 Vectorization Techniques:**

-CountVectorizer

-Converts text into word-frequency vectors

-TF-IDF (Term Frequency–Inverse Document Frequency)

-Gives importance to rare and meaningful words

-----------------------------------------------



**🛠️ Tech Stack & Libraries:**

-pandas
##
-scikit-learn
##
-joblib


-----------------------------------------


**Key imports used in the project:**

-import pandas as pd
######
-import joblib

######
-from sklearn.model_selection import train_test_split
######
-from sklearn.pipeline import Pipeline
#####
-from sklearn.feature_extraction.text import CountVectorizer
#####
-from sklearn.linear_model import LogisticRegression
#####
-from sklearn.naive_bayes import MultinomialNB
######
-from sklearn.svm import LinearSVC
######
-from sklearn.metrics import accuracy_score

--------------------------------------------



**📊 Project Workflow:**
###
-Raw SMS Text
####
     ↓
-Text Vectorization (CountVectorizer / TF-IDF)
####
     ↓
-Train-Test Split
#####
     ↓
-Model Training (NB / LR / SVM)
#####
     ↓
-Model Evaluation
#####
     ↓
-Best Model Saved


------------------------


**🧪 Dataset:**

SMS Spam Collection Dataset
####

Contains labeled SMS messages:
###

spam

ham

-----------------------------------


**📈 Model Evaluation:**

Models are evaluated using Accuracy Score.

Example output:

-Naive Bayes Accuracy: 1.0000
-Logistic Regression Accuracy: 1.0000
-Linear SVM Accuracy: 1.0000

-Best Model Selected: Naive Bayes
-Best Accuracy: 1.0
-Model saved as best_model.pkl


--------------------------------------


**💾 Model Saving:**
###
**The best-performing model is saved using joblib:**
###
-joblib.dump(best_model, "best_model.pkl")
####

**This allows reuse without retraining.**


-------------------------------------------


**▶️ How to Run the Project:**

  1️⃣ Clone the Repository
#####
  git clone https://github.com/your-username/spam-detection-ml.git
  cd spam-detection-ml


---------------------------------------------------------


**2️⃣ Install Dependencies:**
     -pip install -r requirements.txt
------------------------------------
**3️⃣ Train the Model:**
    - python SpamDetection.py
-------------------------------------
**4️⃣ Predict New SMS:**
    - python spam_prediction.py


------------------------------------------     

**📌 Example Prediction:**
     -Message: "Congratulations! You won a free gift"
     -Prediction: Spam
     

Message: "Are we meeting tomorrow?"

Prediction: Ham



--------------------------------------------


**📂 Project Structure:**

   SMS_Detection_ML/
   │
   ├── DATA/
   │       └── spam.csv
   │
   ├── SpamDetection.py
   ├── spam_prediction.py
   ├── best_model.pkl
   ├── README.md


---------------------------------------------


**🎯 Key Learnings:**


   -NLP text preprocessing
###
   -Bag of Words vs TF-IDF
###
   -Supervised classification
###
   -Model comparison and selection
#### 
   -Pipeline-based ML workflow

  

----------------------------------------------

**🧑‍💻 Author**

Anand Upadhyay
##
Aspiring AI / ML Engineer
##
Skilled in Python, Machine Learning, NLP, and Data Science
