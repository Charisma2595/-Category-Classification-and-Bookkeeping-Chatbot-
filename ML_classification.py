import joblib
import numpy as np
from sentence_transformers import SentenceTransformer


model_embedding = SentenceTransformer('all-MiniLM-L6-v2')
model_classification = joblib.load("models/description_classifier.joblib")


def classify_with_ml(description):
    """
    Classify a transaction description using the trained ML model.
    Returns a dict with category, confidence, and review flag.
    """
    try:
        # Encode the description
        embeddings = model_embedding.encode([description], convert_to_tensor=False)
        
        # Get probabilities and prediction
        probabilities = model_classification.predict_proba(embeddings)[0]
        confidence = max(probabilities)
        predicted_category = model_classification.classes_[np.argmax(probabilities)]
        
        # Threshold for classification (adjustable)
        if confidence < 0.7:  # Matches your earlier manual review threshold
            return {
                "category": predicted_category,
                "confidence": float(confidence),
                "needs_review": True
            }
        
        return {
            "category": predicted_category,
            "confidence": float(confidence),
            "needs_review": False
        }
    except Exception as e:
        print(f"Error classifying '{description}': {e}")
        return {
            "category": "Unclassified",
            "confidence": 0.0,
            "needs_review": True
        }

if __name__ == "__main__":
    test_descriptions = [
        "GOOGLE CLOUDTRAININGr",
        "ADOBE CREATIVE CLOUD",
        "FACEBOOK ADS",
        "AMZN UK PRIME",
        "MSFT AZURE",
        "helloo"
    ]
    
    print("Testing ML Classifier:")
    for desc in test_descriptions:
        result = classify_with_ml(desc)
        print(f"{desc} -> Category: {result['category']}, Confidence: {result['confidence']:.2f}, Needs Review: {result['needs_review']}")