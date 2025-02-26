import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as F_sum

# Define the Shared Folder Path
SHARED_FOLDER_PATH = "/Users/a21997/PycharmProjects/RecommendationSystem/data"
FILE_NAMES = ["clients.csv", "products.csv", "stocks.csv", "stores.csv", "transactions.csv"]

# Initialize Spark Session
spark = SparkSession.builder.appName("Recommendation System").config("spark.driver.memory", "8g").getOrCreate()


def load_data():
    """
    Load all CSV files into a dictionary of Pandas DataFrames.
    Returns:
        data_frames (dict): A dictionary containing Pandas DataFrames.
    """
    data_frames = {}

    for file_name in FILE_NAMES:
        file_path = os.path.join(SHARED_FOLDER_PATH, file_name)

        if os.path.exists(file_path):
            print(f"Loading {file_name}...")

            # Use chunksize for efficient memory usage
            chunks = pd.read_csv(file_path, chunksize=100000)
            data_frames[file_name.split(".")[0]] = pd.concat(chunks, ignore_index=True)
        else:
            print(f"⚠️ File not found: {file_name}")

    return data_frames


def clean_data(data_frames):
    """
    Perform data cleaning operations on each dataset.

    Args:
        data_frames (dict): A dictionary containing Pandas DataFrames.

    Returns:
        cleaned_data (dict): A dictionary containing cleaned Pandas DataFrames.
    """

    # Remove duplicates
    for name, df in data_frames.items():
        df.drop_duplicates(inplace=True)
        print(f"{name} removed duplicates")

    # Clean `clients` dataset
    if 'clients' in data_frames:
        clients_df = data_frames['clients']
        clients_df = clients_df[~clients_df['ClientGender'].isin(['C', 'N'])]  # Remove invalid genders
        clients_df['ClientGender'].fillna('U', inplace=True)  # Fill missing gender with 'U'
        clients_df = clients_df[(clients_df['Age'] >= 0) & (clients_df['Age'] <= 100)]  # Remove unreasonable ages
        data_frames['clients'] = clients_df

    # Clean `products` dataset - Check inconsistency
    if 'products' in data_frames:
        conflicts = data_frames['products'].groupby('ProductID')[['Category', 'FamilyLevel1', 'FamilyLevel2']].nunique()
        conflicts = conflicts[(conflicts > 1).any(axis=1)]
        print(f"⚠️ Number of inconsistent ProductIDs in products dataset: {len(conflicts)}")

    # Clean `stocks` dataset
    if 'stocks' in data_frames:
        stocks_df = data_frames['stocks']
        stocks_df = stocks_df[stocks_df['Quantity'] >= 0]  # Remove negative values
        invalid_products = stocks_df[~stocks_df['ProductID'].isin(data_frames['products']['ProductID'])]
        print(f"⚠️ Invalid ProductIDs in stock: {len(invalid_products)}")
        data_frames['stocks'] = stocks_df

    # Clean `stores` dataset - Check StoreCountry consistency
    if 'stores' in data_frames and 'stocks' in data_frames:
        invalid_countries = data_frames['stores'][
            ~data_frames['stores']['StoreCountry'].isin(data_frames['stocks']['StoreCountry'])]
        print(f"⚠️ Invalid StoreCountries in stores: {len(invalid_countries)}")

    # Clean `transactions` dataset - Check foreign key completeness
    if 'transactions' in data_frames:
        transactions_df = data_frames['transactions']

        # Debugging: Print column names
        print("\n🔍 Debug: Transactions DataFrame Columns")
        print(transactions_df.columns.tolist())

        invalid_clients = transactions_df[~transactions_df['ClientID'].isin(data_frames['clients']['ClientID'])]
        invalid_products = transactions_df[~transactions_df['ProductID'].isin(data_frames['products']['ProductID'])]
        invalid_stores = transactions_df[~transactions_df['StoreID'].isin(data_frames['stores']['StoreID'])]

        print(f"⚠️ Invalid ClientIDs in transactions: {len(invalid_clients)}")
        print(f"⚠️ Invalid ProductIDs in transactions: {len(invalid_products)}")
        print(f"⚠️ Invalid StoreIDs in transactions: {len(invalid_stores)}")

        transactions_df = transactions_df[
            (transactions_df['SalesNetAmountEuro'] >= 0) & (transactions_df['Quantity'] >= 0)]
        # Ensure 'SaleTransactionDate' exists
        if 'SaleTransactionDate' not in transactions_df.columns:
            raise KeyError("❌ 'SaleTransactionDate' column is missing after loading!")
        transactions_df['SaleTransactionDate'] = pd.to_datetime(transactions_df['SaleTransactionDate'])
        data_frames['transactions'] = transactions_df

    return data_frames


def convert_to_spark(data_frames):
    """
    Convert cleaned Pandas DataFrames to PySpark DataFrames.

    Args:
        data_frames (dict): A dictionary containing Pandas DataFrames.

    Returns:
        spark_data (dict): A dictionary containing PySpark DataFrames.
    """
    spark_data = {}
    for key, df in data_frames.items():
        spark_data[key] = spark.createDataFrame(df)

    # Rename conflicting column
    spark_data['stocks'] = spark_data['stocks'].withColumnRenamed("Quantity", "StockQuantity")

    return spark_data


def merge_data(spark_data):
    """
    Merge different PySpark DataFrames into one final dataset.

    Args:
        spark_data (dict): A dictionary containing PySpark DataFrames.

    Returns:
        merged_spark (DataFrame): A merged PySpark DataFrame.
    """
    merged_spark = (
        spark_data['transactions']
            .join(spark_data['clients'], "ClientID", "left")
            .join(spark_data['products'], "ProductID", "left")
            .join(spark_data['stores'], "StoreID", "left")
            .join(spark_data['stocks'], on=["StoreCountry", "ProductID"], how="left")
    )

    return merged_spark


def analyze_data(merged_spark):
    """
    Perform exploratory data analysis on the merged dataset.

    Args:
        merged_spark (DataFrame): The merged PySpark DataFrame.
    """
    print("\n🔍 Merged Data Preview:")
    merged_spark.show(5)

    print("\n📌 Schema of Merged Data:")
    merged_spark.printSchema()

    print(f"\n📊 Total Rows: {merged_spark.count()}")

    print("\n📈 Summary Statistics:")
    merged_spark.describe().show()

    print("\n🔍 Checking Missing Values:")
    missing_values = merged_spark.select([F_sum(col(c).isNull().cast("int")).alias(c) for c in merged_spark.columns])
    missing_values.show()


def save_merged_data(merged_spark, output_path="/content/cleaned_merged_data.csv"):
    """
    Save the final merged dataset as a CSV file.

    Args:
        merged_spark (DataFrame): The merged PySpark DataFrame.
        output_path (str): File path to save the merged dataset.
    """
    merged_spark.toPandas().to_csv(output_path, index=False)
    print(f"\n✅ Merged dataset saved at: {output_path}")


if __name__ == "__main__":
    # Load Data
    data_frames = load_data()

    # Clean Data
    cleaned_data = clean_data(data_frames)

    # Convert to Spark DataFrames
    spark_data = convert_to_spark(cleaned_data)

    # Merge Data
    merged_spark = merge_data(spark_data)

    # Analyze Data
    analyze_data(merged_spark)

    # Save Merged Data
    save_merged_data(merged_spark)