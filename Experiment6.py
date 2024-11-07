import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def preprocess_crime_data(df):
    original_shape = df.shape
    preprocessing_stats = {
        'original_rows': original_shape[0],
        'missing_values': {},
        'outliers_removed': {},
    }
    
    missing_values = df.isnull().sum()
    preprocessing_stats['missing_values'] = missing_values.to_dict()
    
    if df['cr'].isnull().any():
        df['cr'].fillna(df['cr'].median(), inplace=True)
    if df['sr'].isnull().any():
        df['sr'].fillna(df['sr'].median(), inplace=True)
    
    if df['Continent'].isnull().any():
        df['Continent'].fillna(df['Continent'].mode()[0], inplace=True)
    if df['Country'].isnull().any():
        df['Country'].fillna('Unknown', inplace=True)
    
    numerical_cols = ['cr', 'sr']
    

    df_zscore = df.copy()
    df_iqr = df.copy()
    

    for col in numerical_cols:
        z_scores = np.abs(stats.zscore(df_zscore[col]))
        outliers_zscore = z_scores > 3
        preprocessing_stats['outliers_removed'][f'{col}_zscore'] = sum(outliers_zscore)
        
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_iqr = (df[col] < lower_bound) | (df[col] > upper_bound)
        preprocessing_stats['outliers_removed'][f'{col}_iqr'] = sum(outliers_iqr)
        
        df_iqr.loc[df_iqr[col] < lower_bound, col] = lower_bound
        df_iqr.loc[df_iqr[col] > upper_bound, col] = upper_bound
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
   
    sns.boxplot(data=df[numerical_cols], ax=axes[0,0])
    axes[0,0].set_title('Original Distribution')
    
  
    sns.boxplot(data=df_zscore[numerical_cols], ax=axes[0,1])
    axes[0,1].set_title('After Z-score Filtering')
    
   
    sns.boxplot(data=df_iqr[numerical_cols], ax=axes[1,0])
    axes[1,0].set_title('After IQR Filtering')
    
    
    sns.scatterplot(data=df, x='cr', y='sr', ax=axes[1,1])
    axes[1,1].set_title('CR vs SR Relationship')
    
    plt.tight_layout()
    
    # Create outlier summary
    outlier_summary = pd.DataFrame({
        'Method': ['Z-score', 'IQR'],
        'CR_Outliers': [preprocessing_stats['outliers_removed']['cr_zscore'],
                       preprocessing_stats['outliers_removed']['cr_iqr']],
        'SR_Outliers': [preprocessing_stats['outliers_removed']['sr_zscore'],
                       preprocessing_stats['outliers_removed']['sr_iqr']]
    })
    
    return df_iqr, outlier_summary, preprocessing_stats


df = pd.read_csv('crime index.csv')


processed_df, outlier_summary, stats = preprocess_crime_data(df)


print("\nPreprocessing Statistics:")
print(f"Original number of rows: {stats['original_rows']}")
print("\nMissing Values:")
for col, count in stats['missing_values'].items():
    print(f"{col}: {count}")

print("\nOutlier Summary:")
print(outlier_summary)


print("\nProcessed Data Statistics:")
print(processed_df.describe())