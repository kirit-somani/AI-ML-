import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class ChurnAnalysis:
    def __init__(self, file_path):
        """Initialize the ChurnAnalysis class with the data file path."""
        self.file_path = file_path
        self.df = None
        
    def load_data(self):
        """Load the data from CSV file and perform initial examination."""
        try:
            self.df = pd.read_csv(self.file_path)
            print("\nData Loading Summary:")
            print(f"Number of rows: {self.df.shape[0]}")
            print(f"Number of columns: {self.df.shape[1]}")
            print("\nFirst few rows of the dataset:")
            print(self.df.head())
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def examine_data_structure(self):
        """Examine the structure of the dataset."""
        print("\nData Structure Analysis:")
        print("\nData Types:")
        print(self.df.dtypes)
        
        print("\nMissing Values Summary:")
        missing_values = self.df.isnull().sum()
        print(missing_values[missing_values > 0])
        
        print("\nBasic Statistics:")
        print(self.df.describe())

    def analyze_categorical_columns(self):
        """Analyze categorical columns and their relationship with churn."""
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        
        print("\nCategorical Columns Analysis:")
        for column in categorical_columns:
            if column != 'customerID':  # Skip customer ID
                print(f"\nDistribution of {column}:")
                distribution = self.df[column].value_counts(normalize=True)
                print(distribution)
                
                # Create cross-tabulation with Churn
                if column != 'Churn':
                    churn_relation = pd.crosstab(
                        self.df[column], 
                        self.df['Churn'], 
                        normalize='index'
                    )
                    print(f"\nChurn Rate by {column}:")
                    print(churn_relation)

    def analyze_numerical_columns(self):
        """Analyze numerical columns and their relationship with churn."""
        numerical_columns = self.df.select_dtypes(include=['int64', 'float64']).columns
        
        print("\nNumerical Columns Analysis:")
        for column in numerical_columns:
            print(f"\nStatistics for {column}:")
            print(self.df.groupby('Churn')[column].describe())
            
            # Detect outliers using IQR method
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.df[
                (self.df[column] < (Q1 - 1.5 * IQR)) | 
                (self.df[column] > (Q3 + 1.5 * IQR))
            ]
            print(f"\nNumber of outliers in {column}: {len(outliers)}")

    def handle_missing_values(self):
        """Handle missing values in the dataset."""
        print("\nHandling Missing Values:")
        
        # For numerical columns
        numerical_columns = self.df.select_dtypes(include=['int64', 'float64']).columns
        if len(numerical_columns) > 0:
            imp = IterativeImputer(random_state=42)
            self.df[numerical_columns] = imp.fit_transform(self.df[numerical_columns])
        
        # For categorical columns
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        for column in categorical_columns:
            if column != 'customerID':  # Skip customer ID
                self.df[column].fillna(self.df[column].mode()[0], inplace=True)
        
        print("Missing values after handling:")
        print(self.df.isnull().sum()[self.df.isnull().sum() > 0])

    def handle_outliers(self):
        """Handle outliers in numerical columns using capping method."""
        numerical_columns = self.df.select_dtypes(include=['int64', 'float64']).columns
        
        print("\nHandling Outliers:")
        for column in numerical_columns:
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            # Cap the outliers
            self.df[column] = np.where(
                self.df[column] < (Q1 - 1.5 * IQR),
                Q1 - 1.5 * IQR,
                np.where(
                    self.df[column] > (Q3 + 1.5 * IQR),
                    Q3 + 1.5 * IQR,
                    self.df[column]
                )
            )
            print(f"Outliers handled in {column}")

    def visualize_data(self):
        """Create visualizations for key insights."""
        plt.figure(figsize=(15, 10))
        
        # Churn Distribution
        plt.subplot(2, 2, 1)
        sns.countplot(data=self.df, x='Churn')
        plt.title('Churn Distribution')
        
        # Tenure vs Churn
        plt.subplot(2, 2, 2)
        sns.boxplot(data=self.df, x='Churn', y='tenure')
        plt.title('Tenure by Churn Status')
        
        # Monthly Charges vs Churn
        plt.subplot(2, 2, 3)
        sns.boxplot(data=self.df, x='Churn', y='MonthlyCharges')
        plt.title('Monthly Charges by Churn Status')
        
        # Contract Type vs Churn
        plt.subplot(2, 2, 4)
        sns.countplot(data=self.df, x='Contract', hue='Churn')
        plt.title('Churn by Contract Type')
        
        plt.tight_layout()
        plt.show()
def main():
    # Initialize the analysis
    analysis = ChurnAnalysis("D:\WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    # Execute analysis steps
    if analysis.load_data():
        analysis.examine_data_structure()
        analysis.analyze_categorical_columns()
        analysis.analyze_numerical_columns()
        analysis.handle_missing_values()
        analysis.handle_outliers()
        analysis.visualize_data()
    else:
        print("Analysis could not be completed due to data loading error.")

if __name__ == "__main__":
    main()
