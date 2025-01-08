import pandas as pd
def universal_frequency_encoding(df, columns_to_encode):
    for col in columns_to_encode:
        freq_encoding = df[col].value_counts(normalize=True)  # Get frequencies
        df[f'{col}_FreqEnc'] = df[col].map(freq_encoding)  # Map frequencies to the column
    return df