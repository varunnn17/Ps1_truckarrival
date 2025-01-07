# import numpy as np
# import pandas as pd

# def preprocess_and_engineer_features(df):
#     """
#     Preprocesses and engineers features for the given DataFrame.
#     Args:
#         df (pd.DataFrame): Input DataFrame with raw features.
#     Returns:
#         pd.DataFrame: DataFrame with engineered features.
#     """

#     # Step 1: Convert PlannedDateTime and ArrivedDateTime to datetime format
#     df['PlannedDateTime'] = pd.to_datetime(df['PlannedDateTime'], format='%d/%m/%Y %H:%M', errors='coerce')

#     print("Data types after conversion:")
#     print(df.dtypes)

#     # Step 2: Extract Day of Week, Hour, and Month from PlannedDateTime
#     df['PlannedDay'] = df['PlannedDateTime'].dt.day
#     df['PlannedHour'] = df['PlannedDateTime'].dt.hour
#     df['DayOfWeek'] = df['PlannedDateTime'].dt.weekday  # 0 = Monday, 6 = Sunday
#     df['Month'] = df['PlannedDateTime'].dt.month
#     df['HourOfDay'] = df['PlannedDateTime'].dt.hour  # Hour in 24-hour format

#     # Step 3: Encode cyclical features for DayOfWeek, Hour, and Month
#     df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
#     df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
#     df['Hour_sin'] = np.sin(2 * np.pi * df['PlannedHour'] / 24)
#     df['Hour_cos'] = np.cos(2 * np.pi * df['PlannedHour'] / 24)
#     df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
#     df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

#     # Step 4: Frequency Encoding for categorical variables
#     for col in ['Ordertype', 'CustomerID', 'RelationID', 'CarrierID']:
#         freq_encoding = df[col].value_counts(normalize=True).to_dict()
#         df[f'{col}_FreqEnc'] = df[col].map(freq_encoding)

#     # Step 5: One-Hot Encoding for Ordertype
#     df = pd.get_dummies(df, columns=['Ordertype'], prefix='Ordertype')

#     # Step 6: Drop unnecessary columns
#     df.drop(columns=['PlannedDateTime', 'ArrivedDateTime', 'DayOfWeek', 'Month', 'PlannedHour'], inplace=True)

#     # Step 7: Adding Carrier Status based on NumberOfOrders
#     carrier_order_sum = df.groupby('CarrierID')['NumberOfOrders'].sum().reset_index()
#     carrier_order_sum_sorted = carrier_order_sum.sort_values(by='NumberOfOrders', ascending=False)
#     top_43_carriers = carrier_order_sum_sorted.head(43)['CarrierID'].tolist()

#     df['CarrierStatus'] = df['CarrierID'].apply(
#         lambda x: 'BusyCarrier' if x in top_43_carriers else 'NonBusyCarrier'
#     )
#     df['CarrierStatus_Encoded'] = df['CarrierStatus'].map({'BusyCarrier': 1, 'NonBusyCarrier': 0})

#     # Drop the original CarrierStatus column
#     df.drop(columns=['CarrierStatus'], inplace=True)

#     required_columns = [
#     'RelationID_FreqEnc', 'HourOfDay', 'Hour_cos',
#     'Ordertype_FreqEnc', 'Hour_sin', 'CarrierStatus_Encoded',
#     'CustomerID_FreqEnc', 'Month_cos', 'DayOfWeek_sin'
# ]
#     df = df[required_columns]
#     return df

# # Example usage:
# # df = pd.read_csv("input.csv")
# # processed_df = preprocess_and_engineer_features(df)
# # print(processed_df.head())


########################################################################################################## 
                           ####Creating a pre processor pickle file####
##########################################################################################################



import numpy as np
import pandas as pd

def preprocess_and_engineer_features(df):
    """
    Preprocesses and engineers features for the given DataFrame.
    Args:
        df (pd.DataFrame): Input DataFrame with raw features.
    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """

    # Step 1: Convert PlannedDateTime and ArrivedDateTime to datetime format
    df['PlannedDateTime'] = pd.to_datetime(df['PlannedDateTime'], format='%d/%m/%Y %H:%M', errors='coerce')

    print("Data types after conversion:")
    print(df.dtypes)

    # Step 2: Extract Day of Week, Hour, and Month from PlannedDateTime
    df['PlannedDay'] = df['PlannedDateTime'].dt.day
    df['PlannedHour'] = df['PlannedDateTime'].dt.hour
    df['DayOfWeek'] = df['PlannedDateTime'].dt.weekday  # 0 = Monday, 6 = Sunday
    df['Month'] = df['PlannedDateTime'].dt.month
    df['HourOfDay'] = df['PlannedDateTime'].dt.hour  # Hour in 24-hour format

    # Step 3: Encode cyclical features for DayOfWeek, Hour, and Month
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    df['Hour_sin'] = np.sin(2 * np.pi * df['PlannedHour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['PlannedHour'] / 24)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    # Step 4: Frequency Encoding for categorical variables
    for col in ['Ordertype', 'CustomerID', 'RelationID', 'CarrierID']:
        freq_encoding = df[col].value_counts(normalize=True).to_dict()
        df[f'{col}_FreqEnc'] = df[col].map(freq_encoding)

    # Step 5: One-Hot Encoding for Ordertype
    df = pd.get_dummies(df, columns=['Ordertype'], prefix='Ordertype')

    # Step 6: Drop unnecessary columns
    df.drop(columns=['PlannedDateTime', 'ArrivedDateTime', 'DayOfWeek', 'Month', 'PlannedHour'], inplace=True)

    # Step 7: Adding Carrier Status based on NumberOfOrders
    carrier_order_sum = df.groupby('CarrierID')['NumberOfOrders'].sum().reset_index()
    carrier_order_sum_sorted = carrier_order_sum.sort_values(by='NumberOfOrders', ascending=False)
    top_43_carriers = carrier_order_sum_sorted.head(43)['CarrierID'].tolist()

    df['CarrierStatus'] = df['CarrierID'].apply(
        lambda x: 'BusyCarrier' if x in top_43_carriers else 'NonBusyCarrier'
    )
    df['CarrierStatus_Encoded'] = df['CarrierStatus'].map({'BusyCarrier': 1, 'NonBusyCarrier': 0})

    # Drop the original CarrierStatus column
    df.drop(columns=['CarrierStatus'], inplace=True)

    required_columns = [
        'RelationID_FreqEnc', 'HourOfDay', 'Hour_cos',
        'Ordertype_FreqEnc', 'Hour_sin', 'CarrierStatus_Encoded',
        'CustomerID_FreqEnc', 'Month_cos', 'DayOfWeek_sin'
    ]
    df = df[required_columns]
    return df

# Save the function as a pickle
if __name__ == "__main__":
    # Example for saving this function into a pickle file
    import pickle

    # Pickle the preprocess function
    with open('preprocessing_pipeline.pkl', 'wb') as f:
        pickle.dump(preprocess_and_engineer_features, f)

    print("Preprocessing function pickled successfully.")



