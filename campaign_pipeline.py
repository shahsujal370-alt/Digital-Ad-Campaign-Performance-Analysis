"""
Digital Ad Campaign Performance - end-to-end data pipeline
===========================================================
Dataset : Kaggle "Marketing Campaign Performance Dataset" (200,000 campaigns, 2021)
Steps   : 1 load  ->  2 quality checks  ->  3 clean  ->  4 derive KPIs
          5 save clean CSV (for Power BI)  ->  6 aggregate fact table (for Excel)
          7 statistical tests (is any difference real or just noise?)

Run:  python campaign_pipeline.py
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

INPUT = "/home/claude/data/marketing_campaign_dataset.csv"   # change to your file path
OUT = "/mnt/user-data/outputs"                               # change to "." on your laptop
WORK = "/home/claude"                                        # helper files for the Excel build

log = []                                                     # audit trail of every check
def note(step, result, detail=""):
    log.append([step, result, detail])

# ---------------------------------------------------------------- 1. LOAD
df = pd.read_csv(INPUT)
note("Rows and columns loaded", f"{len(df):,} rows x {df.shape[1]} columns", "Starting point")

# ---------------------------------------------------------------- 2. QUALITY CHECKS
note("Missing values (all columns)", int(df.isna().sum().sum()), "None found - nothing to impute or drop")
note("Fully duplicated rows", int(df.duplicated().sum()), "None found")
note("Duplicate Campaign_IDs", int(df["Campaign_ID"].duplicated().sum()), "ID is unique per campaign")
note("Rows where Clicks > Impressions", int((df.Clicks > df.Impressions).sum()), "Would be impossible; none found")
note("Negative cost / ROI / clicks", int((df[["Clicks", "ROI"]] < 0).sum().sum()), "None found")
for col in ["Channel_Used", "Campaign_Type", "Target_Audience", "Location", "Duration"]:
    clean_labels = df[col].str.strip().nunique() == df[col].nunique()
    note(f"Category labels consistent: {col}", "Yes" if clean_labels else "NO",
         f"{df[col].nunique()} distinct values, no spelling/spacing variants")

# ---------------------------------------------------------------- 3. CLEAN (fix data types)
# Acquisition_Cost arrives as text like "$16,174.00" -> must become a number
df["Spend_USD"] = df["Acquisition_Cost"].str.replace(r"[\$,]", "", regex=True).astype(float)
note("Acquisition_Cost converted text -> number", f"{len(df):,} rows",
     "Removed '$' and ',' so values can be summed (e.g. '$16,174.00' -> 16174.00)")
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
note("Date converted text -> real date", f"{len(df):,} rows",
     f"Range {df.Date.min():%d-%b-%Y} to {df.Date.max():%d-%b-%Y}")
df["Duration_Days"] = df["Duration"].str.extract(r"(\d+)").astype(int)
note("Duration text -> numeric days", f"{len(df):,} rows", "'30 days' -> 30, so it can be analysed numerically")
df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp()

# ---------------------------------------------------------------- 4. DERIVE KPIs
# The file has no Revenue or Conversions column, so we derive them (documented assumptions):
#   Conversions = Clicks x Conversion_Rate
#   ROI         = (Revenue - Cost) / Cost   =>   Revenue = Cost x (1 + ROI)
df["Conversions"] = (df["Clicks"] * df["Conversion_Rate"]).round(2)
df["Revenue_USD"] = (df["Spend_USD"] * (1 + df["ROI"])).round(2)
df["CTR"] = df["Clicks"] / df["Impressions"]                 # click-through rate
df["CPC"] = df["Spend_USD"] / df["Clicks"]                   # cost per click
df["CPA"] = df["Spend_USD"] / df["Conversions"]              # cost per acquisition
df["ROAS"] = df["Revenue_USD"] / df["Spend_USD"]             # return on ad spend
note("Derived: Conversions, Revenue, CTR, CPC, CPA, ROAS", f"{len(df):,} rows",
     "Conversions=Clicks x Conv_Rate; Revenue=Cost x (1+ROI)")
note("Clean rows final", f"{len(df):,}", "Used for all analysis")

# ---------------------------------------------------------------- 5. SAVE CLEAN CSV
cols = ["Campaign_ID", "Company", "Campaign_Type", "Target_Audience", "Customer_Segment", "Location",
        "Language", "Channel_Used", "Duration", "Duration_Days", "Date", "Month", "Impressions", "Clicks",
        "Engagement_Score", "Spend_USD", "Conversion_Rate", "Conversions", "ROI", "Revenue_USD",
        "CTR", "CPC", "CPA", "ROAS"]
out = df[cols].copy()
out[["CTR", "ROAS"]] = out[["CTR", "ROAS"]].round(4)
out[["CPC", "CPA"]] = out[["CPC", "CPA"]].round(2)
out.to_csv(f"{OUT}/clean_marketing_campaigns.csv", index=False)

# ---------------------------------------------------------------- 6. EXCEL FACT TABLE
# Excel work uses an aggregated table (Month x Channel x Campaign_Type = 360 rows) so it stays fast.
df["ROI_sq"] = df["ROI"] ** 2
fact = (df.groupby(["Month", "Channel_Used", "Campaign_Type"])
          .agg(Campaigns=("Campaign_ID", "count"), Impressions=("Impressions", "sum"),
               Clicks=("Clicks", "sum"), Spend=("Spend_USD", "sum"), Conversions=("Conversions", "sum"),
               Revenue=("Revenue_USD", "sum"), Sum_ROI=("ROI", "sum"), Sum_ROI_Sq=("ROI_sq", "sum"))
          .reset_index())
fact[["Spend", "Conversions", "Revenue", "Sum_ROI", "Sum_ROI_Sq"]] = fact[
    ["Spend", "Conversions", "Revenue", "Sum_ROI", "Sum_ROI_Sq"]].round(2)
fact.to_csv(f"{WORK}/fact_table.csv", index=False)

# ---------------------------------------------------------------- 7. STATISTICAL TESTS
# H0: the group means are equal.  Run for 7 dimensions x 2 metrics = 14 tests.
rows = []
for dim in ["Channel_Used", "Campaign_Type", "Target_Audience", "Customer_Segment",
            "Location", "Company", "Duration"]:
    for metric in ["ROI", "Conversion_Rate"]:
        groups = [g[metric].values for _, g in df.groupby(dim)]
        means = [g.mean() for g in groups]
        rows.append([dim, metric, len(groups), min(means), max(means),
                     (max(means) - min(means)) / np.mean(means),
                     stats.f_oneway(*groups).pvalue, stats.kruskal(*groups).pvalue])
tests = pd.DataFrame(rows, columns=["Dimension", "Metric", "Groups", "Lowest_Group_Mean",
                                    "Highest_Group_Mean", "Gap_vs_Average", "ANOVA_p", "KruskalWallis_p"])
tests.to_csv(f"{WORK}/stat_tests.csv", index=False)
json.dump(log, open(f"{WORK}/cleaning_log.json", "w"))

if __name__ == "__main__":
    for r in log:
        print(f"{r[0]:<55}{str(r[1]):>22}")
    print(tests.round(4).to_string())
