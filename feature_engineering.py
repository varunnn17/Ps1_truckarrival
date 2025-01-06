import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder



def preprocess_and_engineer(data):
    """
    Takes raw input data as a dictionary and applies feature engineering.
    Args:
        data (dict): Input data from the form.

    Returns:
        pd.DataFrame: Processed data ready for prediction.
    """
    df = pd.DataFrame([data])
    df['PlannedDateTime'] = pd.to_datetime(df['PlannedDateTime'], format='%Y-%m-%dT%H:%M')

    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%dT%H:%M')
    
    # Step 2: Extract Day of the Week as numbers (0 = Monday, 6 = Sunday) from 'PlannedDateTime'
    df['DayOfWeekNumber'] = df['PlannedDateTime'].dt.weekday  # 0 = Monday, 6 = Sunday
    
    # Step 3: Extract Hour of the Day (24-hour format) from 'PlannedDateTime'
    df['HourOfDay'] = df['PlannedDateTime'].dt.hour  # Hour in 24-hour format
    
    # Step 4: Extract Month from 'PlannedDateTime' (numeric)
    df['Month'] = df['PlannedDateTime'].dt.month  # Extract the month (1=January, 2=February, ...)
    
    # Step 5: Create a 'Quarter' feature based on the 'Month'
    def get_quarter(month):
        if 1 <= month <= 3:
            return 1  # Q1: January - March
        elif 4 <= month <= 6:
            return 2  # Q2: April - June
        elif 7 <= month <= 9:
            return 3  # Q3: July - September
        else:
            return 4  # Q4: October - December
    
    # Apply the logic to create the 'Quarter' column
    df['Quarter'] = df['Month'].apply(get_quarter)
    
    # Step 6: Apply label encoding to the 'TimeOfDay' column (if needed)
    def categorize_time(hour):
        if 6 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 16:
            return "Afternoon"
        elif 16 <= hour < 22:
            return "Evening"
        else:
            return "Night"
    
    # Create a new column 'TimeOfDay' based on 'HourOfDay'
    df["TimeOfDay"] = df["HourOfDay"].apply(categorize_time)
    
    # Apply label encoding to 'TimeOfDay'
    label_encoder = LabelEncoder()
    df['TimeOfDay_Encoded'] = label_encoder.fit_transform(df['TimeOfDay'])
    
    # Drop the original 'TimeOfDay' column
    df.drop(columns=['TimeOfDay'], inplace=True)
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeekNumber'] / 7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeekNumber'] / 7)
    
    # Cyclic encoding for HourOfDay (0 to 23)
    df['Hour_sin'] = np.sin(2 * np.pi * df['HourOfDay'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['HourOfDay'] / 24)
    
    # Cyclic encoding for Month (1 to 12)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    # Cyclic encoding for Quarter (1 to 4)
    df['Quarter_sin'] = np.sin(2 * np.pi * df['Quarter'] / 4)
    df['Quarter_cos'] = np.cos(2 * np.pi * df['Quarter'] / 4)

    columns_to_encode = ['Ordertype', 'CustomerID', 'RelationID', 'CarrierID']

    # Perform frequency encoding
    for col in columns_to_encode:
        freq_encoding = df[col].value_counts(normalize=True)  # Get frequencies
        df[f'{col}_FreqEnc'] = df[col].map(freq_encoding)  # Map frequencies to the column
    
    aaa = pd.DataFrame(df.groupby('CarrierID')['NumberOfOrders'].sum())
    temp_df = aaa.sort_values(by = "NumberOfOrders" , ascending = False).reset_index()
    
    #25 percent of 171 is 43
    #df_ident
    #Now adding the new feature to the dataframe
    # Identifying the top 43 carriers as "Busy"
    top_43_carriers = temp_df.head(43)['CarrierID'].tolist()
    
    # Creating a new column to label carriers as "BusyCarrier" or "NonBusyCarrier"
    df['CarrierStatus'] = df['CarrierID'].apply(lambda x: 'BusyCarrier' if x in top_43_carriers else 'NonBusyCarrier')
    
    #Encoding - >
    #BusyCarrier -> 1
    #NonBusyCarrier -> 0
    df['CarrierStatus_Encoded'] = df['CarrierStatus'].map({'BusyCarrier': 1, 'NonBusyCarrier': 0})
    df['DayType'] = df['DayOfWeekNumber'].apply(lambda x: 'Weekday' if x in [0,1,2,3,4] else 'Weekend')
 
    df['DayType_Weekend'] = df['DayType'].map({'Weekend': 1, 'Weekday': 0})
    df.drop(columns=['Date','Trip Nr','PlannedDateTime', 'PlannedDay','PlannedHour','Ordertype','CustomerID','RelationID','CarrierID','CarrierStatus', 'DayType'], inplace=True)
    selected_features = [
    'RelationID_FreqEnc', 'HourOfDay', 'Hour_cos',
    'Ordertype_FreqEnc', 'Hour_sin', 'CarrierStatus_Encoded',
    'CustomerID_FreqEnc', 'Month_cos', 'DayOfWeek_sin'
    ]

    testing_data_final = df[selected_features]

    return testing_data_final


