# Find-S Algorithm Implementation

def find_s(examples):
    """
    Find-S algorithm implementation for learning the most specific hypothesis.
    Args:
        examples: list of tuples (attributes, label)
            attributes: list of attribute values
            label: 'Yes' for positive examples, 'No' for negative
    Returns:
        hypothesis: list representing the final hypothesis
    """
    # Step 1: Initialize hypothesis with the first positive example
    hypothesis = None
    for attributes, label in examples:
        if label.strip().lower() == 'yes':
            hypothesis = attributes.copy()  # ensure a copy
            break
    if hypothesis is None:
        raise ValueError("No positive example found in training data!")

    # Step 2: For each example, update the hypothesis for positive examples
    for attributes, label in examples:
        if label.strip().lower() == 'yes':
            for i in range(len(hypothesis)):
                if hypothesis[i] != attributes[i]:
                    hypothesis[i] = '?'  # generalize
    return hypothesis

def test_find_s():
    """
    Simple unit test for the Find-S algorithm.
    Checks for correctness and expected output.
    """
    dataset = [
        (["Sunny", "Warm", "Normal", "Strong", "Warm", "Same"], "Yes"),
        (["Sunny", "Warm", "High", "Strong", "Warm", "Same"], "Yes"),
        (["Rainy", "Cold", "High", "Strong", "Warm", "Change"], "No"),
        (["Sunny", "Warm", "High", "Strong", "Cool", "Change"], "Yes")
    ]
    expected = ['Sunny', 'Warm', '?', 'Strong', '?', '?']
    result = find_s(dataset)
    assert isinstance(result, list), "Find-S should return a list."
    assert len(result) == 6, "Hypothesis should match attribute length."
    assert result == expected, f"Expected {expected}, got {result}"
    print("Find-S test passed.")

if __name__ == "__main__":
    # Training data: (attribute list, label)
    dataset = [
        (["Sunny", "Warm", "Normal", "Strong", "Warm", "Same"], "Yes"),
        (["Sunny", "Warm", "High", "Strong", "Warm", "Same"], "Yes"),
        (["Rainy", "Cold", "High", "Strong", "Warm", "Change"], "No"),
        (["Sunny", "Warm", "High", "Strong", "Cool", "Change"], "Yes")
    ]

    final_hypothesis = find_s(dataset)
    print("Final Hypothesis:", final_hypothesis)
    # Run unit test
    test_find_s()
