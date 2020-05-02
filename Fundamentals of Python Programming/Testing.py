







df.columns
y = df["Income"]

# Single covariates:
covariates = [] # Empty list
for column in df.columns: # Select each column in the dataframe
    covariates.append(column)
covariates.remove("Income")

# Build the individual datasets
df.columns
    # Build the intercept-only model
intercept = pd.DataFrame(1, index=np.arange(1,30137), columns=np.arange(1))
# Build the datasets via a for loop:
datasets = []
for column in covariates: # For each covariate
    datasets.append(sm.add_constant(df.loc[:,column])) # Add the intercept to each covariate
datasets


# Build the models using another for loop
results = []
for dataset in datasets:
    results.append(sm.MNLogit(y, dataset).fit()) # Fit a multinomial logistic regression model


# Show the model summaries, no, it's not iterable
results[0].summary()
results[1].summary()
results[2].summary()
results[3].summary()
results[4].summary()
results[5].summary()
results[6].summary()
results[7].summary()
results[8].summary()
results[9].summary()
results[10].summary()
results[11].summary()
results[12].summary()
results[13].summary()










