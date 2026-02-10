# =====================================================
# Statistical Business Analysis (FINAL VERSION)
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# =====================================================
# 1. LOAD DATASET
# =====================================================
df = pd.read_csv(r"C:\Users\dhans\Downloads\archive (8)\Retail_Sales (1).csv")

# Standardize column names
df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()

print("\nCOLUMNS IN DATASET:")
print(df.columns)

# =====================================================
# 2. IDENTIFY IMPORTANT COLUMNS (SAFE METHOD)
# =====================================================
# Sales column (mandatory)
sales_col = [c for c in df.columns if "sale" in c or "revenue" in c]
if not sales_col:
    raise Exception("No sales column found in dataset.")
sales_col = sales_col[0]

# Region column
region_col = [c for c in df.columns if "region" in c or "area" in c]
region_col = region_col[0] if region_col else None

# Product column
product_col = [c for c in df.columns if "product" in c or "category" in c]
product_col = product_col[0] if product_col else None

print("\nDETECTED COLUMNS")
print("Sales:", sales_col)
print("Region:", region_col)
print("Product:", product_col)

# =====================================================
# 3. DESCRIPTIVE STATISTICS
# =====================================================
print("\nDESCRIPTIVE STATISTICS")
print(df[sales_col].describe())

mean_sales = df[sales_col].mean()
median_sales = df[sales_col].median()
mode_sales = df[sales_col].mode()[0]
std_sales = df[sales_col].std()

print("\nSales Summary")
print(f"Mean: {mean_sales:.2f}")
print(f"Median: {median_sales:.2f}")
print(f"Mode: {mode_sales:.2f}")
print(f"Standard Deviation: {std_sales:.2f}")

# =====================================================
# 4. DISTRIBUTION ANALYSIS
# =====================================================
sns.histplot(df[sales_col], kde=True)
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.show()

shapiro_test = stats.shapiro(df[sales_col])
print("\nShapiro-Wilk Test p-value:", shapiro_test.pvalue)

# =====================================================
# 5. CORRELATION ANALYSIS (NUMERIC ONLY)
# =====================================================
numeric_df = df.select_dtypes(include=np.number)
print("\nCorrelation Matrix")
print(numeric_df.corr())

sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# =====================================================
# 6. HYPOTHESIS TESTING
# =====================================================

# Test 1: One Sample T-Test
t_stat, p_val_1 = stats.ttest_1samp(df[sales_col], mean_sales)
print("\nOne Sample T-Test p-value:", p_val_1)

# Test 2: Independent T-Test (Region-wise)
if region_col:
    regions = df[region_col].unique()
    if len(regions) >= 2:
        g1 = df[df[region_col] == regions[0]][sales_col]
        g2 = df[df[region_col] == regions[1]][sales_col]
        t_stat, p_val_2 = stats.ttest_ind(g1, g2)
        print("Independent T-Test p-value:", p_val_2)

# Test 3: ANOVA (Product-wise)
if product_col:
    groups = [grp[sales_col].values for _, grp in df.groupby(product_col)]
    if len(groups) > 1:
        anova = stats.f_oneway(*groups)
        print("ANOVA Test p-value:", anova.pvalue)

# =====================================================
# 7. CONFIDENCE INTERVAL (95%)
# =====================================================
confidence = 0.95
n = len(df[sales_col])
sem = stats.sem(df[sales_col])

ci = stats.t.interval(confidence, n - 1, mean_sales, sem)
margin_error = (ci[1] - ci[0]) / 2

print("\n95% CONFIDENCE INTERVAL")
print(f"Average Sales: {mean_sales:.0f} ± {margin_error:.0f}")

# =====================================================
# 8. FINAL REPORT
# =====================================================
print("\nSTATISTICAL ANALYSIS REPORT")
print(f"Mean Sales: {mean_sales:.0f}")
print(f"95% Confidence Interval: ± {margin_error:.0f}")

if p_val_1 < 0.05:
    print("Sales are statistically significant.")
else:
    print("Sales are not statistically significant.")
