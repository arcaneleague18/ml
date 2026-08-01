import pandas as pd
try:
    from pomegranate import BayesianNetwork
except ImportError:
    raise ImportError("pomegranate is required for Bayesian Network. Install via pip install pomegranate.")

def build_dataset():
    """
    Create and preprocess the loan dataset for Bayesian Network.
    Returns:
        pd.DataFrame: DataFrame with discretized categorical features for Bayesian Network.
    """
    # Step 1: Create dataset
    df = pd.DataFrame([
        ['LP001002', 5849, 130.0, 'Y'],
        ['LP001003', 4583, 126.0, 'N'],
        ['LP001005', 3000, 66.0, 'Y'],
        ['LP001006', 2583, 120.0, 'N'],
        ['LP001008', 6000, 141.0, 'Y'],
    ], columns=['Loan_ID', 'ApplicantIncome', 'LoanAmount', 'Loan_Status'])

    # Drop Loan_ID (irrelevant for prediction)
    df = df.drop('Loan_ID', axis=1)

    # Discretize ApplicantIncome (Low / Medium / High)
    df['Income_cat'] = pd.cut(df['ApplicantIncome'],
                              bins=[0, 3000, 5000, 7000],
                              labels=['Low', 'Medium', 'High'])

    # Discretize LoanAmount (Small / Medium / Large)
    df['Loan_cat'] = pd.cut(df['LoanAmount'],
                            bins=[0, 100, 130, 200],
                            labels=['Small', 'Medium', 'Large'])

    # Final categorical dataset
    df = df[['Income_cat', 'Loan_cat', 'Loan_Status']]
    return df

def train_bayesian_network(df: pd.DataFrame) -> BayesianNetwork:
    """
    Build a Bayesian Network model from the processed dataset.
    Args:
        df (pd.DataFrame): DataFrame with categorical features.
    Returns:
        BayesianNetwork: Trained Bayesian Network model.
    """
    model = BayesianNetwork.from_samples(df, algorithm='exact')
    return model

def print_model_structure(model: BayesianNetwork):
    """
    Print the structure (edges) of the Bayesian Network.
    Args:
        model (BayesianNetwork): Trained model.
    """
    print("\nBayesian Network Structure:")
    for edge in model.structure:
        print(edge)

def main():
    # Build dataset
    df = build_dataset()

    print("Final dataset used for Bayesian Network:")
    print(df)

    # Build Bayesian Network
    model = train_bayesian_network(df)
    print_model_structure(model)

    # Predicting the class for each row
    predictions = model.predict(df)

    print("\nPredictions (Bayesian Network):")
    print(predictions)

def test_bayes_discretization():
    """
    Basic test to ensure discretization works as expected and output DataFrame is correct.
    """
    df = build_dataset()
    assert list(df.columns) == ['Income_cat', 'Loan_cat', 'Loan_Status'], "Columns mismatch after discretization."
    # Test that no NA values in categories
    assert df['Income_cat'].isnull().sum() == 0, "Income_cat has NaNs."
    assert df['Loan_cat'].isnull().sum() == 0, "Loan_cat has NaNs."
    print("test_bayes_discretization passed.")

def test_bayes_predictions_shape():
    """
    Test that Bayesian Network predicts the same number of rows as in the dataset.
    """
    df = build_dataset()
    model = train_bayesian_network(df)
    predictions = model.predict(df)
    assert len(predictions) == len(df), "Prediction shape mismatch with input rows."
    print("test_bayes_predictions_shape passed.")

if __name__ == "__main__":
    main()
    # Run basic tests
    test_bayes_discretization()
    test_bayes_predictions_shape()
