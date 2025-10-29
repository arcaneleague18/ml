# Find-S Algorithm Implementation

def find_s(examples):
    """
    examples: list of tuples (attributes, label)
    attributes: list of attribute values
    label: 'Yes' for positive examples, 'No' for negative
    """

    # Step 1: Initialize hypothesis with the first positive example
    for attributes, label in examples:
        if label.lower() == 'yes':
            hypothesis = attributes.copy()
            break
    else:
        raise ValueError("No positive example found in training data!")

    # Step 2: For each example, update the hypothesis
    for attributes, label in examples:
        if label.lower() == 'yes':
            for i in range(len(hypothesis)):
                if hypothesis[i] != attributes[i]:
                    hypothesis[i] = '?'  # generalize
                # ignore negative examples

    return hypothesis



# Training data: (attribute list, label)
dataset = [
    (['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same'], 'Yes'),
    (['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same'], 'Yes'),
    (['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change'], 'No'),
    (['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change'], 'Yes')
]

final_hypothesis = find_s(dataset)
print("Final Hypothesis:", final_hypothesis)
