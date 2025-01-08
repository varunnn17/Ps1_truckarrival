import pandas as pd
import numpy as np

def universal_time_based_features(df, time_columns, datetime_format='%d/%m/%Y %H:%M'):
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M', errors='coerce')
    for col in time_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format=datetime_format, errors='coerce')
    for col in time_columns:
        if col in df.columns:
            df['HourOfDay'] = df[col].dt.hour.fillna(0).astype(int)
            df['DayOfWeekNumber'] = df[col].dt.weekday.fillna(0).astype(int)
            df['Month'] = df[col].dt.month.fillna(0).astype(int)
            df['Quarter'] = df[col].dt.quarter.fillna(0).astype(int)
            df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeekNumber'] / 7)
            df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeekNumber'] / 7)
            df['Hour_sin'] = np.sin(2 * np.pi * df['HourOfDay'] / 24)
            df['Hour_cos'] = np.cos(2 * np.pi * df['HourOfDay'] / 24)
            df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
            df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
            df['Quarter_sin'] = np.sin(2 * np.pi * df['Quarter'] / 4)
            df['Quarter_cos'] = np.cos(2 * np.pi * df['Quarter'] / 4)
    return df