
def train_pipeline(ingest_data, clean_data, model_train, evaluate_model):
    """
    Args:
        ingest_data: Class
        clean_data: Class
        model_train: Class
        evaluate_model : Class
    Returns:
        mae: float,
        mse: float,
        rmse: float,
        r2_score : float
    """
    data = ingest_data.initiate_ingest_data()
    train_arr, test_arr, _ = clean_data.clean_data_and_transform(data=data)
    X_train, X_test, y_train, y_test, best_model = model_train.initiate_model_training(
        train_arr, test_arr)
    mae, mse, rmse, r2_score = evaluate_model.evaluate_single_model(
        X_train, X_test, y_train, y_test, best_model)
    return mae, mse, rmse, r2_score
