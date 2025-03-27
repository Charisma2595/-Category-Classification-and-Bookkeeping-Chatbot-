import pandas as pd
from regex_classification import classify_with_regex
from ML_classification import classify_with_ml
from llm_classification import classify_with_llm

def classify(transactions):
    """
    Classify a list of (description, amount) tuples using the hybrid approach.
    Returns a list of classification results.
    """
    labels = []
    for description, amount in transactions:
        label = classify_transaction(description, amount)
        labels.append(label)
    return labels

def classify_transaction(description, amount):
    """
    Classify a single transaction using the hybrid approach:
    1. Regex -> 2. LLM (Payroll) -> 3. ML/BERT.
    Returns a dict with category, confidence, needs_review.
    """
    # Step 1: Try regex classification
    label = classify_with_regex(description)
    if label and label != "Unclassified":
        if isinstance(label, str):  
            return {
                "category": label,
                "confidence": 0.9,
                "needs_review": False
            }
        return label  

    # Step 2: Try LLM for Payroll
    llm_result = classify_with_llm(description)
    if isinstance(llm_result, dict) and llm_result["category"] == "Payroll":
        return llm_result
    
    # Step 3: Fall back to ML/BERT
    bert_result = classify_with_ml(description)
    if isinstance(bert_result, str): 
        return {
            "category": bert_result,
            "confidence": 0.7,
            "needs_review": bert_result == "Unclassified"
        }
    return bert_result  

def classify_csv(input_file):
    """
    Read a CSV with 'description' and 'amount', classify each row, and save to output CSV.
    """
    # Read CSV
    df = pd.read_csv(input_file)
    if "description" not in df.columns or "amount" not in df.columns:
        raise ValueError("CSV must contain 'description' and 'amount' columns")

    results = classify(list(zip(df["description"], df["amount"])))

    normalized_results = []
    for r in results:
        if isinstance(r, str):
            normalized_results.append({
                "category": r,
                "confidence": 1.0,
                "needs_review": False
            })
        elif isinstance(r, dict):
            normalized_results.append(r)
        else:
            normalized_results.append({
                "category": "Unclassified",
                "confidence": 0.0,
                "needs_review": True
            })

    # Assign to DataFrame columns
    df["category"] = [r["category"] for r in normalized_results]
    df["confidence"] = [r["confidence"] for r in normalized_results]
    df["needs_review"] = [r["needs_review"] for r in normalized_results]

    # Save to output CSV
    output_file = "test_files/output.csv"
    df.to_csv(output_file, index=False)
    print(f"Classified transactions saved to {output_file}")
    
    return output_file

if __name__ == "__main__":
    classify_csv("test_files/test.csv")