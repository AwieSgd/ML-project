import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


# Создаем датасет с пропусками в колонке message
df_spam_dirty = pd.DataFrame({
    'id': range(1, 21),
    'message': [
        "Congratulations! You've won a free iPhone. Click here to claim.",
        None,  # Пропуск
        "URGENT: Your bank account has been locked. Verify details now.",
        "Please review the attached project report before the meeting.",
        np.nan,  # Пропуск
        "Can you send me the notes from today's lecture?",
        "Claim your exclusive discount voucher code now, limited time!",
        "Mom called, she wants to know if you are coming home this weekend.",
        "Free entry to a luxury vacation resort. Call fast!",
        None,  # Пропуск
        "Earn 5000 dollars working from home with zero experience.",
        "Let's schedule a call on Zoom to discuss the MLOps pipeline.",
        "Your loan application of 50k has been pre-approved. Apply today.",
        "Don't forget to buy groceries: milk, bread, and eggs.",
        "Claim your crypto wallet reward worth 1 Bitcoin right now!",
        np.nan,  # Пропуск
        "Get cheap medications online without prescription.",
        "Could you help me debug this Python script when you have time?",
        "Exclusive offer: Buy one get two free on all items.",
        "Thanks for helping me with the Git repository setup today.",
    ],
    'label': [
        'spam',
        'ham',
        'spam',
        'ham',
        'spam',
        'ham',
        'spam',
        'ham',
        'spam',
        'ham',
        'spam',
        'ham',
        'spam',
        'ham',
        'spam',
        'ham',
        'spam',
        'ham',
        'spam',
        'ham',
    ],
})

df_spam_dirty['message'] = df_spam_dirty['message'].fillna('')
x = df_spam_dirty[["message"]]
y = df_spam_dirty["label"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

text_orig = "message"
text_set = Pipeline(steps=[('tfidf', TfidfVectorizer(max_features=100))])

preprocessor = ColumnTransformer(
    transformers=[
        ('text_set', text_set, text_orig)])

model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('model', GradientBoostingClassifier(random_state=42))])

param_grid = {
    'model__n_estimators': [50, 60, 70],
    'model__learning_rate': [0.01, 0.1, 0.2],
    }

best_pipeline = GridSearchCV(estimator=model_pipeline, param_grid= param_grid, cv=3)
best_pipeline.fit(x_train, y_train)
print(best_pipeline.best_params_, '\n')
print(best_pipeline.best_score_, '\n')
predictions = best_pipeline.predict(x_test)
classification_report = classification_report(y_test, predictions)
print(classification_report, '\n')
comparison = pd.DataFrame({'Real': y_test, 'Predicted': predictions})
print(comparison)
