import numpy as np


def candidate_elimination(concepts, target) :
    '''
    concepts: list of training examples
    target: list of target labels ('Yes'/'No')
    '''

    # Initialize specific hypothesis to the first positive example
    specific_h = concepts [ 0 ].copy ( )
    print ( "Initialization of specific hypothesis:", specific_h )

    # Initialize general hypothesis with the most general hypothesis
    general_h = [ [ '?' for _ in range ( len ( specific_h ) ) ] for _ in range ( len ( specific_h ) ) ]
    print ( "Initialization of general hypothesis:", general_h )

    for i, h in enumerate ( concepts ) :
        if target [ i ] == "Yes" :
            for x in range ( len ( specific_h ) ) :
                if h [ x ] != specific_h [ x ] :
                    specific_h [ x ] = '?'
                    general_h [ x ] [ x ] = '?'
        elif target [ i ] == "No" :
            for x in range ( len ( specific_h ) ) :
                if h [ x ] != specific_h [ x ] :
                    general_h [ x ] [ x ] = specific_h [ x ]
                else :
                    general_h [ x ] [ x ] = '?'
        print ( "\nStep", i + 1 )
        print ( "Instance:", h )
        print ( "Target:", target [ i ] )
        print ( "Specific hypothesis:", specific_h )
        print ( "General hypothesis:", general_h )

    # Remove duplicates
    general_h = [ g for g in general_h if g != [ '?' for _ in range ( len ( specific_h ) ) ] ]
    return specific_h, general_h


# Example Training Data
concepts = np.array ( [
    [ 'Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same' ],
    [ 'Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same' ],
    [ 'Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change' ],
    [ 'Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change' ]
] )

target = np.array ( [ 'Yes', 'Yes', 'No', 'Yes' ] )

# Run Candidate Elimination
s_final, g_final = candidate_elimination ( concepts, target )

print ( "\nFinal Specific Hypothesis:\n", s_final )
print ( "\nFinal General Hypotheses:\n", g_final )
