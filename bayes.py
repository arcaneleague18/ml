import pandas as pd
from pomegranate import BayesianNetwork

# Step 1: Create dataset
df = pd.DataFrame([
    ['LP001002', 5849, 130.0, 'Y'],
    ['LP001003', 4583, 126.0, 'N'],
    ['LP001005', 3000, 66.0, 'Y'],
    ['LP001006', 2583, 120.0, 'N'],
    ['LP001008', 6000, 141.0, 'Y'],
], columns=['Loan_ID', 'ApplicantIncome', 'LoanAmount', 'Loan_Status'])

# Drop Loan_ID
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

print("Final dataset used for Bayesian Network:")
print(df)

# Build Bayesian Network
model = BayesianNetwork.from_samples(df, algorithm='exact')

print("\nBayesian Network Structure:")
for edge in model.structure:
    print(edge)

# Predicting the class for each row
predictions = model.predict(df)

print("\nPredictions (Bayesian Network):")
print(predictions)