from dotenv import load_dotenv
from groq import Groq
import re
import os


load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file. Please set it as GROQ_API_KEY=your_key_here")

groq = Groq(api_key=api_key)

def classify_with_llm(description):
    """
    Classify a transaction description using LLM, restricted to Payroll category.
    Returns a dict with category, confidence, and needs_review for Payroll-related,
    or just "Unclassified" string for non-Payroll.
    """

    payroll_pattern = r"(?i)\b(payroll|salary|wage|payment|employee pay)\b"
    
    # Check if description matches Payroll pattern
    if not re.search(payroll_pattern, description):
        return "Unclassified"
    
    # If Payroll, use LLM to classify/confirm
    prompt = f'''Classify the transaction description into one of these categories:
    (1) Payroll, (2) Unclassified.
    If you can't determine it’s Payroll-related, use "Unclassified".
    Put the category inside <category> </category> tags.
    Description: {description}'''

    try:
        chat_completion = groq.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="deepseek-r1-distill-llama-70b",
            temperature=0.5
        )

        content = chat_completion.choices[0].message.content
        match = re.search(r'<category>(.*)</category>', content, flags=re.DOTALL)
        category = "Unclassified"
        if match:
            category = match.group(1).strip()

        # Confidence-like score (simplified, based on LLM output clarity)
        confidence = 0.95 if category == "Payroll" else 0.60
        
        return {
            "category": category,
            "confidence": confidence,
            "needs_review": confidence < 0.7
        }
    
    except Exception as e:
        print(f"Error classifying '{description}' with LLM: {e}")
        return {
            "category": "Unclassified",
            "confidence": 0.0,
            "needs_review": True
        }

if __name__ == "__main__":
    test_descriptions = [
        "PAYROLL TRANSFER",
        "GOOGLE CLOUDTRAININGr",
        "Salary payment",
        "FACEBOOK ADS",
        "Employee wage adjustment",
        "AMZN UK PRIME"
    ]
    
    print("Testing LLM Classifier (Payroll Only):")
    for desc in test_descriptions:
        result = classify_with_llm(desc)
        if isinstance(result, str):
            print(f"{desc} -> {result}")
        else:
            print(f"{desc} -> Category: {result['category']}, Confidence: {result['confidence']:.2f}, Needs Review: {result['needs_review']}")