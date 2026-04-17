import random
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType
from pyspark.sql.functions import current_timestamp, rand, expr, col
from datetime import datetime, timedelta

def main():
    spark = SparkSession.builder.getOrCreate()
    
    # Retrieve Unity Catalog settings passed as job parameters (or use defaults)
    # In a real environment, you'd pass these via dbutils.widgets
    # or rely on the workspace default catalog/schema.
    # For this example, we assume we want to write to `main.fraud_detection.transactions`
    catalog = "main"
    schema = "fraud_detection_dev"
    table_name = "transactions"
    
    full_table_name = f"{catalog}.{schema}.{table_name}"
    print(f"Generating synthetic data to {full_table_name}")
    
    # Ensure the schema exists
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
    
    # Generate 100,000 synthetic transactions
    num_records = 100000
    
    # Generate base dataframe
    # We will simulate credit card transactions:
    # - transaction_id: string
    # - user_id: int
    # - amount: double
    # - category: string
    # - is_fraud: int (0 or 1)
    
    data = spark.range(0, num_records).selectExpr(
        "cast(id as string) as transaction_id",
        "cast(rand() * 10000 as int) as user_id",
        "round(rand() * 1000, 2) as amount",
        "cast(rand() * 5 as int) as category_id"
    )
    
    # Map category_id to strings and inject some basic fraud rules
    # High amount + specific categories have higher fraud probability
    data = data.withColumn(
        "category",
        expr(
            "CASE WHEN category_id = 0 THEN 'Retail' "
            "WHEN category_id = 1 THEN 'Travel' "
            "WHEN category_id = 2 THEN 'Dining' "
            "WHEN category_id = 3 THEN 'Online Shopping' "
            "ELSE 'Other' END"
        )
    ).drop("category_id")
    
    # Add timestamps (last 30 days)
    data = data.withColumn(
        "timestamp",
        expr(f"current_timestamp() - INTERVAL cast(rand() * 30 as int) DAYS - INTERVAL cast(rand() * 24 as int) HOURS")
    )
    
    # Add fraud labels (synthetic rule: high amount in Online Shopping or Travel is more likely fraud, plus random noise)
    data = data.withColumn(
        "fraud_probability",
        expr("CASE "
             "WHEN amount > 800 AND category IN ('Online Shopping', 'Travel') THEN 0.8 "
             "WHEN amount > 500 THEN 0.3 "
             "ELSE 0.05 END")
    )
    
    data = data.withColumn(
        "is_fraud",
        expr("cast(rand() < fraud_probability as int)")
    ).drop("fraud_probability")
    
    # Write to Delta table using Unity Catalog
    print("Writing to Delta table...")
    data.write \
        .format("delta") \
        .mode("overwrite") \
        .saveAsTable(full_table_name)
        
    print(f"Successfully generated {num_records} records in {full_table_name}")

if __name__ == "__main__":
    main()
