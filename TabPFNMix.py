import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


if __name__ == '__main__':
    # Load your custom dataset instead of the example one
    input_file_path = r"C:\Users\Joseph\OneDrive\文件\Joseph\MIS545\545FinalDataset_FrequencyEncoded.csv"
    df = pd.read_csv(input_file_path)

    # Encode target column
    df['IncidentGrade'] = df['IncidentGrade'].apply(lambda x: 1 if x == 'FalsePositive' else 0)

    # Train-test split
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(df, test_size=0.2, stratify=df['IncidentGrade'], random_state=42)

    # Rename target column to "class"
    train_data = train_data.rename(columns={"IncidentGrade": "class"})
    test_data = test_data.rename(columns={"IncidentGrade": "class"})

    # Define TabPFNMix default hyperparameters with 50 epochs
    tabpfnmix_default = {
        "model_path_classifier": "autogluon/tabpfn-mix-1.0-classifier",
        "model_path_regressor": "autogluon/tabpfn-mix-1.0-regressor",
        "n_ensembles": 3,
        "max_epochs": 35,  # Set 50 epochs
    }

    hyperparameters = {
        "TABPFNMIX": [tabpfnmix_default],
    }

    # Set the target variable's label
    label = "class"

    # Initialize TabularPredictor
    predictor = TabularPredictor(label=label)

    try:
        # Train the TabPFNMix model
        predictor = predictor.fit(
            train_data=train_data,
            hyperparameters=hyperparameters,
            verbosity=3,
        )

        # Predict on test data
        predictions = predictor.predict(test_data.drop(columns=["class"]))
        y_test = test_data["class"]

        # Calculate performance metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        # Print the metrics
        print("\nPerformance Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Display the leaderboard for performance evaluation
        leaderboard = predictor.leaderboard(test_data, display=True)

        print("\nLeaderboard Results:")
        print(leaderboard)

    except Exception as e:
        print(f"Error during training or evaluation: {e}")
