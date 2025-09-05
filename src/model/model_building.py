import numpy as np
import pandas as pd
import os
import pickle
import yaml
import logging
from sklearn.ensemble import RandomForestClassifier

# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load preprocessed train data from CSV."""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def split_features_target(df: pd.DataFrame, target_col="HeartDisease"):
    """Split dataframe into features and target."""
    try:
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        logger.debug("Features and target separated")
        return X, y
    except Exception as e:
        logger.error(f"Error splitting features and target: {e}")
        raise


def train_rf(X_train: np.ndarray, y_train: np.ndarray, n_estimators: int, random_state: int):
    """Train a Random Forest model."""
    try:
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )
        rf_model.fit(X_train, y_train)
        logger.debug("Random Forest model training completed")
        return rf_model
    except Exception as e:
        logger.error('Error during Random Forest training: %s', e)
        raise


def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise


def get_root_directory() -> str:
    """Get the root directory (two levels up from this script's location)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))


def main():
    try:
        # Get root directory
        root_dir = get_root_directory()

        # Load parameters from params.yaml
        params = load_params(os.path.join(root_dir, 'params.yaml'))
        n_estimators = params['model_building']['n_estimators']
        random_state = params['model_building']['random_state']

        # Load preprocessed train dataset
        train_data = load_data(os.path.join(root_dir, 'data/processed/train_processed.csv'))

        # Split into features and target
        X_train, y_train = split_features_target(train_data)

        # Train Random Forest model
        rf_model = train_rf(X_train, y_train, n_estimators, random_state)

        # Save the trained model
        save_model(rf_model, os.path.join(root_dir, 'rf_model.pkl'))

    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
