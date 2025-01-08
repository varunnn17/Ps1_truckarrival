from sklearn.preprocessing import LabelEncoder
import pandas as pd

def universal_label_encoding(df, columns_to_encode):
    for col in columns_to_encode:
        le = LabelEncoder()
        df[col + '_Encoded'] = le.fit_transform(df[col])
    return df
   