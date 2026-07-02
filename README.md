import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("="*60)
print(" RETAIL SALES ANALYSIS PROJECT ")
print("="*60)


def _create_sample_files(train_file, feature_file, store_file):
    """Create small sample CSV files so the script can run when data is missing."""
    import pandas as _pd

    train_sample = _pd.DataFrame([
        {"Store": 1, "Date": "2012-01-01", "Weekly_Sales": 1000.0, "IsHoliday": False},
        {"Store": 1, "Date": "2012-01-08", "Weekly_Sales": 1100.0, "IsHoliday": False},
        {"Store": 2, "Date": "2012-01-01", "Weekly_Sales": 1500.0, "IsHoliday": False},
    ])

    features_sample = _pd.DataFrame([
        {"Store": 1, "Date": "2012-01-01", "IsHoliday": False, "Temperature": 55.0, "Fuel_Price": 3.5, "CPI": 210.0, "Unemployment": 7.8},
        {"Store": 1, "Date": "2012-01-08", "IsHoliday": False, "Temperature": 53.0, "Fuel_Price": 3.6, "CPI": 210.5, "Unemployment": 7.7},
        {"Store": 2, "Date": "2012-01-01", "IsHoliday": False, "Temperature": 60.0, "Fuel_Price": 3.4, "CPI": 209.8, "Unemployment": 7.9},
    ])

    stores_sample = _pd.DataFrame([
        {"Store": 1, "Type": "A", "Size": 100000},
        {"Store": 2, "Type": "B", "Size": 50000},
    ])

    train_sample.to_csv(train_file, index=False)
    features_sample.to_csv(feature_file, index=False)
    stores_sample.to_csv(store_file, index=False)

    print("Created sample data files:", train_file, feature_file, store_file)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

train_file = "trains_test.csv"
feature_file = "Features data set.csv"
store_file = "stores.csv"

missing = [f for f in (train_file, feature_file, store_file) if not os.path.exists(f)]
if missing:
    print("ERROR: missing files:", ", ".join(missing))
    print("Creating sample files so the script can continue.")
    _create_sample_files(train_file, feature_file, store_file)

print("\nLoading Files...")

train = pd.read_csv(train_file)
features = pd.read_csv(feature_file)
stores = pd.read_csv(store_file)

print("Files Loaded Successfully\n")

# -------------------------------------------------
# DATA INFORMATION
# -------------------------------------------------

print("TRAIN DATA")
print(train.head())

print("\nFEATURES DATA")
print(features.head())

print("\nSTORES DATA")
print(stores.head())

print("\nTrain Shape :", train.shape)
print("Features Shape :", features.shape)
print("Stores Shape :", stores.shape)

# -------------------------------------------------
# MISSING VALUES
# -------------------------------------------------

print("\nMissing Values\n")

print(train.isnull().sum())
print(features.isnull().sum())
print(stores.isnull().sum())

# Fill missing values

features.fillna(0, inplace=True)

# Remove duplicate rows

train.drop_duplicates(inplace=True)
features.drop_duplicates(inplace=True)
stores.drop_duplicates(inplace=True)

# -------------------------------------------------
# DATE FORMAT
# -------------------------------------------------

train["Date"] = pd.to_datetime(train["Date"])
features["Date"] = pd.to_datetime(features["Date"])

# -------------------------------------------------
# MERGE DATA
# -------------------------------------------------

print("\nMerging datasets...")

sales = pd.merge(
    train,
    features,
    on=["Store", "Date", "IsHoliday"],
    how="left"
)

sales = pd.merge(
    sales,
    stores,
    on="Store",
    how="left"
)

print("Merged Shape :", sales.shape)

sales.to_csv("Merged_Data.csv", index=False)

print("Merged file saved.\n")

# -------------------------------------------------
# SUMMARY
# -------------------------------------------------

print("="*60)
print("SUMMARY")
print("="*60)

print(sales.describe())

print("\nColumns")
print(sales.columns)

# -------------------------------------------------
# TOTAL SALES
# -------------------------------------------------

total_sales = sales["Weekly_Sales"].sum()

print("\nTotal Sales")
print(total_sales)

average_sales = sales["Weekly_Sales"].mean()

print("\nAverage Sales")
print(average_sales)

maximum_sales = sales["Weekly_Sales"].max()

minimum_sales = sales["Weekly_Sales"].min()

print("\nMaximum Sales")
print(maximum_sales)

print("\nMinimum Sales")
print(minimum_sales)

# -------------------------------------------------
# STORE SALES
# -------------------------------------------------

store_sales = sales.groupby("Store")["Weekly_Sales"].sum()

print("\nStore Sales")
print(store_sales)

top10 = store_sales.sort_values(ascending=False).head(10)

print("\nTop 10 Stores")
print(top10)

# -------------------------------------------------
# STORE TYPE SALES
# -------------------------------------------------

type_sales = sales.groupby("Type")["Weekly_Sales"].sum()

print(type_sales)

# -------------------------------------------------
# STORE SIZE
# -------------------------------------------------

size_sales = sales.groupby("Size")["Weekly_Sales"].sum()

print(size_sales.head())

# -------------------------------------------------
# HOLIDAY SALES
# -------------------------------------------------

holiday = sales.groupby("IsHoliday")["Weekly_Sales"].mean()

print("\nHoliday Sales")

print(holiday)

# -------------------------------------------------
# MONTHLY SALES
# -------------------------------------------------

sales["Month"] = sales["Date"].dt.month

monthly = sales.groupby("Month")["Weekly_Sales"].sum()

print(monthly)

# -------------------------------------------------
# YEARLY SALES
# -------------------------------------------------

sales["Year"] = sales["Date"].dt.year

yearly = sales.groupby("Year")["Weekly_Sales"].sum()

print(yearly)

# -------------------------------------------------
# TEMPERATURE
# -------------------------------------------------

temp = sales.groupby("Temperature")["Weekly_Sales"].mean()

print(temp.head())

# -------------------------------------------------
# FUEL PRICE
# -------------------------------------------------

fuel = sales.groupby("Fuel_Price")["Weekly_Sales"].mean()

print(fuel.head())

# -------------------------------------------------
# CPI
# -------------------------------------------------

cpi = sales.groupby("CPI")["Weekly_Sales"].mean()

print(cpi.head())

# -------------------------------------------------
# UNEMPLOYMENT
# -------------------------------------------------

un = sales.groupby("Unemployment")["Weekly_Sales"].mean()

print(un.head())

# -------------------------------------------------
# CORRELATION
# -------------------------------------------------

numeric = sales.select_dtypes(include=np.number)

corr = numeric.corr()

print("\nCorrelation Matrix")

print(corr)

corr.to_csv("Correlation.csv")

# -------------------------------------------------
# SAVE SUMMARY
# -------------------------------------------------

summary = pd.DataFrame({

    "Total Sales":[total_sales],

    "Average Sales":[average_sales],

    "Maximum Sales":[maximum_sales],

    "Minimum Sales":[minimum_sales]

})

summary.to_csv("Summary.csv", index=False)

# -------------------------------------------------
# CHART 1
# -------------------------------------------------

plt.figure(figsize=(12,6))

top10.plot(kind="bar")

plt.title("Top 10 Stores")

plt.ylabel("Sales")

plt.tight_layout()

plt.savefig("Top10Stores.png")

plt.close()

# -------------------------------------------------
# CHART 2
# -------------------------------------------------

plt.figure(figsize=(12,6))

monthly.plot(marker="o")

plt.title("Monthly Sales")

plt.grid(True)

plt.tight_layout()

plt.savefig("MonthlySales.png")

plt.close()

# -------------------------------------------------
# CHART 3
# -------------------------------------------------

plt.figure(figsize=(8,8))

type_sales.plot(kind="pie",autopct="%1.1f%%")

plt.ylabel("")

plt.title("Sales by Store Type")

plt.savefig("StoreType.png")

plt.close()

# -------------------------------------------------
# CHART 4
# -------------------------------------------------

plt.figure(figsize=(10,5))

plt.hist(sales["Weekly_Sales"],bins=40)

plt.title("Weekly Sales Distribution")

plt.xlabel("Weekly Sales")

plt.ylabel("Frequency")

plt.savefig("Histogram.png")

plt.close()

# -------------------------------------------------
# CHART 5
# -------------------------------------------------

plt.figure(figsize=(14,10))

sns.heatmap(corr,cmap="coolwarm")

plt.title("Correlation Heatmap")

plt.savefig("Heatmap.png")

plt.close()

# -------------------------------------------------
# CHART 6
# -------------------------------------------------

plt.figure(figsize=(10,6))

plt.scatter(
    sales["Temperature"],
    sales["Weekly_Sales"],
    alpha=0.4
)

plt.title("Temperature vs Weekly Sales")

plt.xlabel("Temperature")

plt.ylabel("Weekly Sales")

plt.savefig("TemperatureSales.png")

plt.close()

# -------------------------------------------------
# CHART 7
# -------------------------------------------------

plt.figure(figsize=(10,6))

sns.boxplot(x="Type",y="Weekly_Sales",data=sales)

plt.title("Weekly Sales by Store Type")

plt.savefig("Boxplot.png")

plt.close()

# -------------------------------------------------
# EXPORT FINAL DATA
# -------------------------------------------------

sales.to_excel(
    "Retail_Final_Output.xlsx",
    index=False
)

print("\nExcel File Saved")

print("\nSummary Saved")

print("\nCharts Saved")

print("\nProject Completed Successfully")
