import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# small-text specific imports
from small_text import PoolBasedActiveLearner, SklearnDataset, random_initialization
from small_text.classifiers import SklearnClassifierFactory
from small_text.query_strategies import LeastConfidence

import joblib

# ==========================================
# 1. Load and Prepare the Toy Data
# ==========================================
# We'll use just two categories from the 20 Newsgroups dataset to keep it fast
newsgroups = fetch_20newsgroups(
    subset="train", categories=["rec.sport.baseball", "sci.space"]
)

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
x_features = vectorizer.fit_transform(newsgroups.data)
y_labels = newsgroups.target  # We hide these and only reveal them when "queried"

# Wrap the data in small-text's specific SklearnDataset format
target_labels = np.array([0, 1])
train_dataset = SklearnDataset(x_features, y_labels, target_labels=target_labels)


# ==========================================
# 2. Set up the Active Learner
# ==========================================
# small-text uses a "factory" to generate your model (Naive Bayes in this case)
classifier_factory = SklearnClassifierFactory(MultinomialNB(), num_classes=2)

# Define how the model chooses what to ask you.
# 'LeastConfidence' mathematically picks the items the model is most unsure about.
query_strategy = LeastConfidence()

# Initialize the Active Learner
active_learner = PoolBasedActiveLearner(
    classifier_factory, query_strategy, train_dataset
)


# ==========================================
# 3. Initialize with a Small "Seed"
# ==========================================
# We need to give the model a tiny bit of data to start. Let's randomly pick 10 examples.
np.random.seed(42)

print("======================================================")
print("Initializing with a small random seed of 5 examples...")
print("======================================================")

initial_indices = random_initialization(train_dataset, n_samples=5)
initial_labels = []

for idx in initial_indices:
    print(f"\nTEXT PREVIEW: {newsgroups.data[idx][:300]}...")
    label = input("Label (0 for Baseball/No, 1 for Space/Yes): ")
    initial_labels.append(int(label))
    print("---------------------------------------------")

# "Label" them. In a real scenario, you would manually read and label these 10.
# Here, we just pass the indices, and the active learner grabs the true labels we hid earlier.
active_learner.initialize_data(initial_indices, y_labels[initial_indices])


# ==========================================
# 4. The Active Learning Loop
# ==========================================
num_queries = 3
samples_per_query = 5

print("=============================================")
print("Starting Active Learning Loop...\n")
print("=============================================")

for i in range(num_queries):
    # Step A: The learner calculates uncertainties and asks for the most informative texts
    queried_indices = active_learner.query(num_samples=samples_per_query)

    # Step B: manual labeling
    current_labels = []
    for count, idx in enumerate(queried_indices):
        print(f"\nItem {count+1}/{samples_per_query}")
        print(f"TEXT: {newsgroups.data[idx][:500]}...")

        label = input("Label (0 for Baseball/No, 1 for Space/Yes): ")
        current_labels.append(int(label))
        print("---------------------------------------------")

    # Step C: Feed the answers back to the model so it can retrain itself
    active_learner.update(np.array(current_labels))

    # Step D: Evaluate how well it's doing
    predictions = active_learner.classifier.predict(train_dataset)
    acc = accuracy_score(y_labels, predictions)

    print(
        f"Iteration {i+1} | Total labeled: {len(active_learner.indices_labeled)} | Accuracy on whole pool: {acc:.2%}"
    )


print("\nSaving the model and vectorizer...")

# 1. Extract the underlying scikit-learn model from the small-text wrapper
final_model = active_learner.classifier.model

# 2. Save the model to your directory
joblib.dump(final_model, "relevance_model.pkl")

# 3. Save the vectorizer (CRITICAL!)
joblib.dump(vectorizer, "relevance_vectorizer.pkl")

print("Saved successfully!")
