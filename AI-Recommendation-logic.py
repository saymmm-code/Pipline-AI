import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = {
    'Role': [
        'DevOps Engineer',
        'Data Scientist',
        'Cloud Architect',
        'Software QA Engineer',
        'Frontend Developer'
    ],
    'Skills': [
        'Python Cloud Automation Docker CI/CD Kubernetes Linux',
        'Python Machine Learning SQL Data Analysis Statistics Pandas',
        'Cloud Security Architecture AWS Infrastructure Automation Networking',
        'Python Playwright Automation Testing Postman SQL API QA',
        'JavaScript React HTML CSS Web Design UI UX Frontend'
    ]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['Skills'])

user_inputs = ["Python", "Cloud", "Automation"]
user_profile_str = " ".join(user_inputs)

user_vector = vectorizer.transform([user_profile_str])

similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

df['Similarity_Score'] = similarity_scores

top_recommendations = df.sort_values(by='Similarity_Score', ascending=False).head(3)

print("--- Top 3 Recommendations ---")
for idx, row in top_recommendations.iterrows():
    print(f"Role: {row['Role']} | Match Score: {row['Similarity_Score']:.2f}")