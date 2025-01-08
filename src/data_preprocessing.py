import pandas as pd
import numpy as np
from utils.Date_time import universal_time_based_features
from utils.Label_encode import universal_label_encoding
from utils.Freq_encode import universal_frequency_encoding
import pickle

def preprocess_and_engineer(df):
    time_columns = ['PlannedDateTime']
    df = universal_time_based_features(df, time_columns, datetime_format='%d/%m/%Y %H:%M')
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
    df = universal_label_encoding(df, ['TimeOfDay'])
    columns_to_encode = ['Ordertype', 'CustomerID', 'RelationID', 'CarrierID']
    df = universal_frequency_encoding(df, columns_to_encode)
    aaa = pd.DataFrame(df.groupby('CarrierID')['NumberOfOrders'].sum())
    temp_df = aaa.sort_values(by = "NumberOfOrders" , ascending = False).reset_index()
    #25 percent of 171 is 43
    #df_ident
    #Now adding the new feature to the dataframe
    # Identifying the top 43 carriers as "Busy"
    top_43_carriers = temp_df.head(43)['CarrierID'].tolist()
    # Creating a new column to label carriers as "BusyCarrier" or "NonBusyCarrier"
    df['CarrierStatus'] = df['CarrierID'].apply(lambda x: 'BusyCarrier' if x in top_43_carriers else 'NonBusyCarrier')
    df['CarrierStatus_Encoded'] = df['CarrierStatus'].map({'BusyCarrier': 1, 'NonBusyCarrier': 0})
    df['DayType'] = df['DayOfWeekNumber'].apply(lambda x: 'Weekday' if x in [0,1,2,3,4] else 'Weekend')
    df['DayType_Weekend'] = df['DayType'].map({'Weekend': 1, 'Weekday': 0})

    df.drop(columns=['Date','Trip Nr','PlannedDateTime','Ordertype','CustomerID','RelationID','CarrierID','CarrierStatus', 'DayType', 'TimeOfDay'], inplace=True)
    selected_features = [
    'RelationID_FreqEnc', 'HourOfDay', 'Hour_cos',
    'Ordertype_FreqEnc', 'Hour_sin', 'CarrierStatus_Encoded',
    'CustomerID_FreqEnc', 'Month_cos', 'DayOfWeek_sin'
    ]
    return df[selected_features]

def save_preprocessor(preprocessor, filename='models/preprocessor.pkl'):
    # Pickle the preprocessor function or transformations used for feature engineering
    with open(filename, 'wb') as f:
        pickle.dump(preprocessor, f)





