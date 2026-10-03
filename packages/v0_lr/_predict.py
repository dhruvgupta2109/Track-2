model, features = joblib.load("model.pkl")  # uploaded next to this notebook
test_df = pd.read_csv(TEST_PATH)
X_test = make_features(test_df)[features]
preds = model.predict_proba(X_test)[:, 1]
