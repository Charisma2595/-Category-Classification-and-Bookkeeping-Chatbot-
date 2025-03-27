import re
def classify_with_regex(description):
    regex_patterns = {
        r"UBER RIDE.*": ("Travel", 0.99),
        r"UBER.*": ("Travel", 0.99),
        r"BRITISH GAS.*": ("Utilities", 0.99)
    }
    
    for pattern, (label, confidence) in regex_patterns.items():
        if re.search(pattern, description):
            return {
                "category": label,
                "confidence": confidence,
                "needs_review": confidence < 0.7
            }
    
    return None


if __name__ == "__main__":
    print(classify_with_regex("BRITISH GAS SMART HOME"))
    print(classify_with_regex("UBER TRIP"))
    print(classify_with_regex("GOOGLE ADS"))