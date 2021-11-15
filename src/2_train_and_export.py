import pickle
import pandas as pd
import catboost as ctb

from code.utils.RandomValidator import RandomValidator
from code.utils.properties import cat_features, cols_to_drop, searching_params

if __name__ == '__main__':
    train_df = pd.read_csv('../dataset/train.csv')
    test_df = pd.read_csv('../dataset/test.csv')

    train_df[cat_features] = train_df[cat_features].astype(str)
    test_df[cat_features] = test_df[cat_features].astype(str)

    # Prepare data for training
    X_train = train_df.drop(cols_to_drop, axis=1)
    y_train = train_df.default

    # Compute class multiplier (for imbalanced datasets)
    pos_class_multiplier = len([x for x in y_train if x == 0]) / len([x for x in y_train if x == 1])
    print('Class multiplier for class 1 is', round(pos_class_multiplier, 1))

    fixed_params = {
        'class_weights': (1, pos_class_multiplier)
    }

    ctb_cv_validator = RandomValidator(X_train, y_train, fixed_params=fixed_params, searching_params=searching_params,
                                       num_folds=5, num_iterations=30, granularity=10)
    ctb_cv_validator.run()

    # Fit the best model configuration found
    best_params = ctb_cv_validator.get_best_params()

    final_model = ctb.CatBoostClassifier(**best_params)
    final_model.fit(X_train, y_train, cat_features=[X_train.columns.tolist().index(x) for x in cat_features], )

    # Get prediction probabilities of the test entries
    test_preds = final_model.predict_proba(test_df[X_train.columns])[:, 1]

    prediction_df = pd.DataFrame({'uuid': test_df.uuid, 'pd': test_preds})

    # save prediction file
    prediction_df.to_csv('../output/predictions.csv', index=False)

    # display predictions
    prediction_df.head()

    # Save trained model as pickle file
    with open('final_model.pkl', 'wb') as f:
        pickle.dump(final_model, f)