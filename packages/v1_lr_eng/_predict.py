bundle = joblib.load("model.pkl")  # uploaded next to this notebook
model, features, K = bundle["model"], bundle["features"], bundle["constants"]
test_df = pd.read_csv(TEST_PATH)
X_test = make_features(test_df, K)[features]
preds = model.predict_proba(X_test)[:, 1]
