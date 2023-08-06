
def train_pipeline(ingest_data, clean_data, model_train):
    """
    Args:
        ingest_data: Class
        clean_data: Class
        model_train: Class
    Returns:
        mae: float,
        mse: float,
        rmse: float,
        r2_score : float
    """
    data = ingest_data.initiate_ingest_data()
    train_arr, test_arr, _ = clean_data.clean_data_and_transform(data=data)
    mae, mse, rmse, r2_score = model_train.initiate_model_training(
        train_arr, test_arr)
    return mae, mse, rmse, r2_score
