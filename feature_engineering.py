# # import pandas as pd
# # import numpy as np
# # from sklearn.preprocessing import LabelEncoder



# # def preprocess_and_engineer(data):
# #     """
# #     Takes raw input data as a dictionary and applies feature engineering.
# #     Args:
# #         data (dict): Input data from the form.

# #     Returns:
# #         pd.DataFrame: Processed data ready for prediction.
# #     """
# #     df = pd.DataFrame([data])
# #     df['PlannedDateTime'] = pd.to_datetime(df['PlannedDateTime'], format='%Y-%m-%dT%H:%M')

# #     df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%dT%H:%M')
    
# #     # Step 2: Extract Day of the Week as numbers (0 = Monday, 6 = Sunday) from 'PlannedDateTime'
# #     df['DayOfWeekNumber'] = df['PlannedDateTime'].dt.weekday  # 0 = Monday, 6 = Sunday
    
# #     # Step 3: Extract Hour of the Day (24-hour format) from 'PlannedDateTime'
# #     df['HourOfDay'] = df['PlannedDateTime'].dt.hour  # Hour in 24-hour format
    
# #     # Step 4: Extract Month from 'PlannedDateTime' (numeric)
# #     df['Month'] = df['PlannedDateTime'].dt.month  # Extract the month (1=January, 2=February, ...)
    
# #     # Step 5: Create a 'Quarter' feature based on the 'Month'
# #     def get_quarter(month):
# #         if 1 <= month <= 3:
# #             return 1  # Q1: January - March
# #         elif 4 <= month <= 6:
# #             return 2  # Q2: April - June
# #         elif 7 <= month <= 9:
# #             return 3  # Q3: July - September
# #         else:
# #             return 4  # Q4: October - December
    
# #     # Apply the logic to create the 'Quarter' column
# #     df['Quarter'] = df['Month'].apply(get_quarter)
    
# #     # Step 6: Apply label encoding to the 'TimeOfDay' column (if needed)
# #     def categorize_time(hour):
# #         if 6 <= hour < 12:
# #             return "Morning"
# #         elif 12 <= hour < 16:
# #             return "Afternoon"
# #         elif 16 <= hour < 22:
# #             return "Evening"
# #         else:
# #             return "Night"
    
# #     # Create a new column 'TimeOfDay' based on 'HourOfDay'
# #     df["TimeOfDay"] = df["HourOfDay"].apply(categorize_time)
    
# #     # Apply label encoding to 'TimeOfDay'
# #     label_encoder = LabelEncoder()
# #     df['TimeOfDay_Encoded'] = label_encoder.fit_transform(df['TimeOfDay'])
    
# #     # Drop the original 'TimeOfDay' column
# #     df.drop(columns=['TimeOfDay'], inplace=True)
# #     df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeekNumber'] / 7)
# #     df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeekNumber'] / 7)
    
# #     # Cyclic encoding for HourOfDay (0 to 23)
# #     df['Hour_sin'] = np.sin(2 * np.pi * df['HourOfDay'] / 24)
# #     df['Hour_cos'] = np.cos(2 * np.pi * df['HourOfDay'] / 24)
    
# #     # Cyclic encoding for Month (1 to 12)
# #     df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
# #     df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
# #     # Cyclic encoding for Quarter (1 to 4)
# #     df['Quarter_sin'] = np.sin(2 * np.pi * df['Quarter'] / 4)
# #     df['Quarter_cos'] = np.cos(2 * np.pi * df['Quarter'] / 4)

# #     columns_to_encode = ['Ordertype', 'CustomerID', 'RelationID', 'CarrierID']

# #     # Perform frequency encoding
# #     for col in columns_to_encode:
# #         freq_encoding = df[col].value_counts(normalize=True)  # Get frequencies
# #         df[f'{col}_FreqEnc'] = df[col].map(freq_encoding)  # Map frequencies to the column
    
# #     aaa = pd.DataFrame(df.groupby('CarrierID')['NumberOfOrders'].sum())
# #     temp_df = aaa.sort_values(by = "NumberOfOrders" , ascending = False).reset_index()
    
# #     #25 percent of 171 is 43
# #     #df_ident
# #     #Now adding the new feature to the dataframe
# #     # Identifying the top 43 carriers as "Busy"
# #     top_43_carriers = temp_df.head(43)['CarrierID'].tolist()
    
# #     # Creating a new column to label carriers as "BusyCarrier" or "NonBusyCarrier"
# #     df['CarrierStatus'] = df['CarrierID'].apply(lambda x: 'BusyCarrier' if x in top_43_carriers else 'NonBusyCarrier')
    
# #     #Encoding - >
# #     #BusyCarrier -> 1
# #     #NonBusyCarrier -> 0
# #     df['CarrierStatus_Encoded'] = df['CarrierStatus'].map({'BusyCarrier': 1, 'NonBusyCarrier': 0})
# #     df['DayType'] = df['DayOfWeekNumber'].apply(lambda x: 'Weekday' if x in [0,1,2,3,4] else 'Weekend')
 
# #     df['DayType_Weekend'] = df['DayType'].map({'Weekend': 1, 'Weekday': 0})
# #     df.drop(columns=['Date','Trip Nr','PlannedDateTime', 'PlannedDay','PlannedHour','Ordertype','CustomerID','RelationID','CarrierID','CarrierStatus', 'DayType'], inplace=True)
# #     selected_features = [
# #     'RelationID_FreqEnc', 'HourOfDay', 'Hour_cos',
# #     'Ordertype_FreqEnc', 'Hour_sin', 'CarrierStatus_Encoded',
# #     'CustomerID_FreqEnc', 'Month_cos', 'DayOfWeek_sin'
# #     ]

# #     testing_data_final = df[selected_features]

# #     return testing_data_final

# # import pandas as pd
# # import numpy as np
# # from sklearn.preprocessing import LabelEncoder

# # def preprocess_and_engineer(data):
# #     """
# #     Takes raw input data as a dictionary and applies feature engineering.
# #     Args:
# #         data (dict): Input data from the form or a row from the uploaded CSV.

# #     Returns:
# #         pd.DataFrame: Processed data ready for prediction.
# #     """
# #     # Step 1: Convert input data to DataFrame
# #     df = pd.DataFrame([data])

# #     # Try parsing 'PlannedDateTime' and 'Date' columns in multiple formats
# #     try:
# #         df['PlannedDateTime'] = pd.to_datetime(df['PlannedDateTime'], 
# #                                                 errors='coerce', 
# #                                                 dayfirst=False, 
# #                                                 format='%Y-%m-%dT%H:%M')  # try first format
# #         if df['PlannedDateTime'].isna().any():  # Check for failed parsing
# #             df['PlannedDateTime'] = pd.to_datetime(df['PlannedDateTime'], errors='coerce', format='%m/%d/%Y %H:%M')  # try another format
# #     except Exception as e:
# #         print(f"Error in parsing 'PlannedDateTime': {e}")

# #     try:
# #         df['Date'] = pd.to_datetime(df['Date'], 
# #                                      errors='coerce', 
# #                                      dayfirst=False, 
# #                                      format='%Y-%m-%dT%H:%M')  # try first format
# #         if df['Date'].isna().any():  # Check for failed parsing
# #             df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%m/%d/%Y')  # try another format
# #     except Exception as e:
# #         print(f"Error in parsing 'Date': {e}")

# #     # Step 2: Feature Engineering

# #     # Extract Day of the Week from 'PlannedDateTime' (0 = Monday, 6 = Sunday)
# #     df['DayOfWeekNumber'] = df['PlannedDateTime'].dt.weekday  # 0 = Monday, 6 = Sunday

# #     # Extract Hour of the Day from 'PlannedDateTime'
# #     df['HourOfDay'] = df['PlannedDateTime'].dt.hour  # Hour in 24-hour format

# #     # Extract Month from 'PlannedDateTime' (1 = January, 2 = February, ...)
# #     df['Month'] = df['PlannedDateTime'].dt.month

# #     # Step 3: Create a 'Quarter' feature based on the 'Month'
# #     def get_quarter(month):
# #         if 1 <= month <= 3:
# #             return 1  # Q1: January - March
# #         elif 4 <= month <= 6:
# #             return 2  # Q2: April - June
# #         elif 7 <= month <= 9:
# #             return 3  # Q3: July - September
# #         else:
# #             return 4  # Q4: October - December

# #     # Apply the logic to create the 'Quarter' column
# #     df['Quarter'] = df['Month'].apply(get_quarter)

# #     # Step 4: Categorize 'TimeOfDay' based on 'HourOfDay'
# #     def categorize_time(hour):
# #         if 6 <= hour < 12:
# #             return "Morning"
# #         elif 12 <= hour < 16:
# #             return "Afternoon"
# #         elif 16 <= hour < 22:
# #             return "Evening"
# #         else:
# #             return "Night"

# #     # Create a new column 'TimeOfDay' based on 'HourOfDay'
# #     df["TimeOfDay"] = df["HourOfDay"].apply(categorize_time)

# #     # Step 5: Apply Label Encoding to 'TimeOfDay'
# #     label_encoder = LabelEncoder()
# #     df['TimeOfDay_Encoded'] = label_encoder.fit_transform(df['TimeOfDay'])

# #     # Drop the original 'TimeOfDay' column
# #     df.drop(columns=['TimeOfDay'], inplace=True)

# #     # Step 6: Cyclic encoding for 'DayOfWeek', 'HourOfDay', 'Month', 'Quarter'
# #     df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeekNumber'] / 7)
# #     df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeekNumber'] / 7)
    
# #     # Cyclic encoding for HourOfDay (0 to 23)
# #     df['Hour_sin'] = np.sin(2 * np.pi * df['HourOfDay'] / 24)
# #     df['Hour_cos'] = np.cos(2 * np.pi * df['HourOfDay'] / 24)
    
# #     # Cyclic encoding for Month (1 to 12)
# #     df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
# #     df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
# #     # Cyclic encoding for Quarter (1 to 4)
# #     df['Quarter_sin'] = np.sin(2 * np.pi * df['Quarter'] / 4)
# #     df['Quarter_cos'] = np.cos(2 * np.pi * df['Quarter'] / 4)

# #     # Step 7: Frequency Encoding for certain columns
# #     columns_to_encode = ['Ordertype', 'CustomerID', 'RelationID', 'CarrierID']

# #     for col in columns_to_encode:
# #         freq_encoding = df[col].value_counts(normalize=True)  # Get frequencies
# #         df[f'{col}_FreqEnc'] = df[col].map(freq_encoding)  # Map frequencies to the column

# #     # Step 8: Identify Busy Carriers based on Number of Orders
# #     aaa = pd.DataFrame(df.groupby('CarrierID')['NumberOfOrders'].sum())
# #     temp_df = aaa.sort_values(by="NumberOfOrders", ascending=False).reset_index()
# #     top_43_carriers = temp_df.head(43)['CarrierID'].tolist()
    
# #     df['CarrierStatus'] = df['CarrierID'].apply(lambda x: 'BusyCarrier' if x in top_43_carriers else 'NonBusyCarrier')
# #     df['CarrierStatus_Encoded'] = df['CarrierStatus'].map({'BusyCarrier': 1, 'NonBusyCarrier': 0})

# #     # Step 9: DayType Encoding (Weekday vs Weekend)
# #     df['DayType'] = df['DayOfWeekNumber'].apply(lambda x: 'Weekday' if x in [0,1,2,3,4] else 'Weekend')
# #     df['DayType_Weekend'] = df['DayType'].map({'Weekend': 1, 'Weekday': 0})

# #     # Step 10: Final Feature Selection
# #     selected_features = [
# #         'RelationID_FreqEnc', 'HourOfDay', 'Hour_cos',
# #         'Ordertype_FreqEnc', 'Hour_sin', 'CarrierStatus_Encoded',
# #         'CustomerID_FreqEnc', 'Month_cos', 'DayOfWeek_sin'
# #     ]

# #     testing_data_final = df[selected_features]

# #     return testing_data_final



# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder

# def preprocess_and_engineer(df):
#     """
#     Preprocesses and engineers features from the raw input dataframe.
#     Args:
#         df (pd.DataFrame): Input dataframe.
#     Returns:
#         pd.DataFrame: Processed dataframe ready for prediction.
#     """


#     # Convert 'PlannedDateTime' by combining 'Planned Date' and 'Planned Time'
#     df["PlannedDateTime"] = pd.to_datetime(
#         df["Planned Date"] + " " + df["Planned Time"],
#         format="%d/%m/%Y %I:%M:%S %p"
#     )
    
#     # Convert 'ArrivedDateTime' by combining 'Arrival Date' and 'Arrival Time'
#     df["ArrivedDateTime"] = pd.to_datetime(
#         df["Arrival Date"] + " " + df["Arrival Time"],
#         format="%d/%m/%Y %I:%M:%S %p"
#     )


#     # Step 4: Feature Engineering
#     # Extract Day of the Week as numbers (0 = Monday, 6 = Sunday) from 'PlannedDateTime'
#     df['DayOfWeekNumber'] = df['PlannedDateTime'].dt.weekday  # 0 = Monday, 6 = Sunday
    
#     # Extract Hour of the Day (24-hour format) from 'PlannedDateTime'
#     df['HourOfDay'] = df['PlannedDateTime'].dt.hour  # Hour in 24-hour format
    
#     # Extract Month from 'PlannedDateTime' (numeric)
#     df['Month'] = df['PlannedDateTime'].dt.month  # Extract the month (1=January, 2=February, ...)
    
#     # Step 5: Create a 'Quarter' feature based on the 'Month'
#     def get_quarter(month):
#         if 1 <= month <= 3:
#             return 1  # Q1: January - March
#         elif 4 <= month <= 6:
#             return 2  # Q2: April - June
#         elif 7 <= month <= 9:
#             return 3  # Q3: July - September
#         else:
#             return 4  # Q4: October - December
    
#     # Apply the logic to create the 'Quarter' column
#     df['Quarter'] = df['Month'].apply(get_quarter)
    
#     # Step 6: Create 'TimeOfDay' feature based on 'HourOfDay'
#     def categorize_time(hour):
#         if 6 <= hour < 12:
#             return "Morning"
#         elif 12 <= hour < 16:
#             return "Afternoon"
#         elif 16 <= hour < 22:
#             return "Evening"
#         else:
#             return "Night"
    
#     df["TimeOfDay"] = df["HourOfDay"].apply(categorize_time)
    
#     # Apply label encoding to 'TimeOfDay'
#     label_encoder = LabelEncoder()
#     df['TimeOfDay_Encoded'] = label_encoder.fit_transform(df['TimeOfDay'])
    
#     # Drop the original 'TimeOfDay' column
#     df.drop(columns=['TimeOfDay'], inplace=True)
    
#     # Step 7: Create cyclic encodings for DayOfWeek, HourOfDay, Month, and Quarter
#     df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeekNumber'] / 7)
#     df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeekNumber'] / 7)
    
#     # Cyclic encoding for HourOfDay (0 to 23)
#     df['Hour_sin'] = np.sin(2 * np.pi * df['HourOfDay'] / 24)
#     df['Hour_cos'] = np.cos(2 * np.pi * df['HourOfDay'] / 24)
    
#     # Cyclic encoding for Month (1 to 12)
#     df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
#     df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
#     # Cyclic encoding for Quarter (1 to 4)
#     df['Quarter_sin'] = np.sin(2 * np.pi * df['Quarter'] / 4)
#     df['Quarter_cos'] = np.cos(2 * np.pi * df['Quarter'] / 4)
    
#     # Step 8: Perform Frequency Encoding for categorical columns
#     columns_to_encode = ['Ordertype', 'CustomerID', 'CarrierID']
    
#     for col in columns_to_encode:
#         freq_encoding = df[col].value_counts(normalize=True)  # Get frequencies
#         df[f'{col}_FreqEnc'] = df[col].map(freq_encoding)  # Map frequencies to the column
    
#     # Step 9: Adding Carrier Status based on NumberOfOrders
#     aaa = pd.DataFrame(df.groupby('CarrierID')['NumberOfOrders'].sum())
#     temp_df = aaa.sort_values(by="NumberOfOrders", ascending=False).reset_index()
    
#     # Identifying the top 43 carriers as "Busy"
#     top_43_carriers = temp_df.head(43)['CarrierID'].tolist()
    
#     # Creating a new column to label carriers as "BusyCarrier" or "NonBusyCarrier"
#     df['CarrierStatus'] = df['CarrierID'].apply(lambda x: 'BusyCarrier' if x in top_43_carriers else 'NonBusyCarrier')
    
#     # Encoding - BusyCarrier -> 1, NonBusyCarrier -> 0
#     df['CarrierStatus_Encoded'] = df['CarrierStatus'].map({'BusyCarrier': 1, 'NonBusyCarrier': 0})
    
#     # Step 10: Adding Day Type (Weekend/Weekday)
#     df['DayType'] = df['DayOfWeekNumber'].apply(lambda x: 'Weekday' if x in [0,1,2,3,4] else 'Weekend')
#     df['DayType_Weekend'] = df['DayType'].map({'Weekend': 1, 'Weekday': 0})
    
#     # Drop unnecessary columns
#     df.drop(columns=['Date', 'Trip Nr', 'Planned Date', 'Planned Time', 'Arrival Date', 'Arrival Time', 'CarrierID', 'CarrierStatus', 'DayType'], inplace=True)
    
#     # Step 11: Select the final set of features for prediction
#     selected_features = [
#         'Ordertype_FreqEnc', 'HourOfDay', 'Hour_cos', 'Ordertype_FreqEnc', 'Hour_sin', 
#         'CarrierStatus_Encoded', 'CustomerID_FreqEnc', 'Month_cos', 'DayOfWeek_sin'
#     ]
    
#     # Return the processed dataframe with selected features
#     testing_data_final = df[selected_features]
    
#     return testing_data_final



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

# Example usage:
# df = pd.read_csv("input.csv")
# processed_df = preprocess_and_engineer_features(df)
# print(processed_df.head())






