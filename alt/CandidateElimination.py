import numpy as np

def candidate_elimination(concepts, target):
    '''
    Candidate Elimination Algorithm for concept learning.
    Args:
        concepts: list/array of training examples (each row is an instance)
        target: list/array of target labels ('Yes'/'No')
    Returns:
        specific_h: the final specific hypothesis
        general_h: list of final general hypotheses
    '''
    # Initialize specific hypothesis to the first positive example
    specific_h = None
    for idx, label in enumerate(target):
        if label.strip().lower() == "yes":
            specific_h = concepts[idx].copy()
            break
    if specific_h is None:
        raise ValueError("No positive example found in training data!")

    print("Initialization of specific hypothesis:", specific_h)

    # Initialize general hypothesis with the most general hypothesis
    general_h = [[ '?' for _ in range(len(specific_h)) ] for _ in range(len(specific_h))]
    print("Initialization of general hypothesis:", general_h)

    for i, instance in enumerate(concepts):
        if target[i].strip().lower() == "yes":
            for x in range(len(specific_h)):
                if instance[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'
        elif target[i].strip().lower() == "no":
            for x in range(len(specific_h)):
                if instance[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'
        print(f"\nStep {i + 1}")
        print("Instance:", instance)
        print("Target:", target[i])
        print("Specific hypothesis:", specific_h)
        print("General hypothesis:", general_h)

    # Remove overly general hypotheses (all '?') and duplicates
    most_general = ['?' for _ in range(len(specific_h))]
    general_h = [g for g in general_h if g != most_general]
    # Remove duplicates
    unique_general_h = []
    for g in general_h:
        if g not in unique_general_h:
            unique_general_h.append(g)
    general_h = unique_general_h
    return specific_h, general_h

def test_candidate_elimination():
    """
    Simple unit test for candidate_elimination algorithm.
    """
    concepts = np.array([
        [ 'Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same' ],
        [ 'Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same' ],
        [ 'Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change' ],
        [ 'Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change' ]
    ])
    target = np.array([ 'Yes', 'Yes', 'No', 'Yes' ])
    s_final, g_final = candidate_elimination(concepts, target)
    assert isinstance(s_final, (list, np.ndarray)), "Output specific_h should be a list or np.ndarray"
    assert isinstance(g_final, list), "Output general_h should be a list"
    assert len(s_final) == concepts.shape[1], "specific_h length must match concept attributes"
    print("\nTest passed: candidate_elimination basic output checks.")

if __name__ == "__main__":
    # Example Training Data
    concepts = np.array([
        [ 'Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same' ],
        [ 'Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same' ],
        [ 'Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change' ],
        [ 'Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change' ]
    ])

    target = np.array([ 'Yes', 'Yes', 'No', 'Yes' ])

    # Run Candidate Elimination
    s_final, g_final = candidate_elimination(concepts, target)

    print("\nFinal Specific Hypothesis:\n", s_final)
    print("\nFinal General Hypotheses:\n", g_final)

    # Run test
    test_candidate_elimination()
