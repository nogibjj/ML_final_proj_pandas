# This script preprocesses the test data by joining all the tables and creating new features
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os, gc, joblib
from pprint import pprint
import lightgbm as lgb
from sklearn import metrics
from functools import reduce
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    StratifiedGroupKFold,
)
from contextlib import suppress

pathway = ""


# helper function taken from discussions
def set_table_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    for col in df.columns:
        # Cast Transform DPD (Days past due, P) and Transform Amount (A) as Float64
        if col[-1] in ("P", "A"):
            df = df.with_columns(pl.col(col).cast(pl.Float64).alias(col))
        if col[-4:-1] in ("_sum"):
            df = df.with_columns(pl.col(col).cast(pl.Float64).alias(col))
        if col[-4:-1] in ("_max"):
            df = df.with_columns(pl.col(col).cast(pl.Float64).alias(col))
    return df


def convert_strings(df: pl.DataFrame) -> pl.DataFrame:
    for col in df.columns:
        if df[col].dtype == pl.Utf8:
            df = df.with_columns(pl.col(col).cast(pl.Categorical))
    return df


# checking what percentage of each column is missing
def missing_values(df, threshold=0.0):
    for col in df.columns:
        decimal = (pd.isnull(test[col]).sum()) / (len(test[col]))
        if decimal > threshold:
            print(f"{col}: {decimal}")


# Impute numeric columns with the median and cat with mode
def imputer(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].fillna(df[col].median())
        if df[col].dtype.name in ["category", "object"] and df[col].isnull().any():
            mode_without_nan = df[col].dropna().mode().values[0]
            df[col] = df[col].fillna(mode_without_nan)
    return df


test_basetable = pl.read_csv(pathway + "csv_files/test/test_base.csv")
test_static = pl.concat(
    [
        pl.read_csv(pathway + "csv_files/test/test_static_0_0.csv").pipe(
            set_table_dtypes
        ),
        pl.read_csv(pathway + "csv_files/test/test_static_0_1.csv").pipe(
            set_table_dtypes
        ),
        pl.read_csv(pathway + "csv_files/test/test_static_0_2.csv").pipe(
            set_table_dtypes
        ),
    ],
    how="vertical_relaxed",
)
test_static_cb = pl.read_csv(pathway + "csv_files/test/test_static_cb_0.csv").pipe(
    set_table_dtypes
)
test_person_1 = pl.read_csv(pathway + "csv_files/test/test_person_1.csv").pipe(
    set_table_dtypes
)
test_credit_bureau_b_2 = pl.read_csv(
    pathway + "csv_files/test/test_credit_bureau_b_2.csv"
).pipe(set_table_dtypes)


# files tagged as "other" are depth=1
test_other_1 = pl.read_csv(pathway + "csv_files/test/test_other_1.csv").pipe(
    set_table_dtypes
)

test_credit_bureau_b_1 = pl.read_csv(
    pathway + "csv_files/test/test_credit_bureau_b_1.csv"
).pipe(set_table_dtypes)

test_deposit_1 = pl.read_csv(pathway + "csv_files/test/test_deposit_1.csv").pipe(
    set_table_dtypes
)

test_basetable = test_basetable.with_columns(pl.col("date_decision").cast(pl.Date))


# Aggregate columns for depth=2 files so that familial features are not duplicated
test_person_1_feats_1 = test_person_1.group_by("case_id").agg(
    pl.col("mainoccupationinc_384A").sum().alias("mainoccupationinc_384A_sum")
)

# num_group1=0 represents the person who applied for the loan
test_person_1_feats_2 = (
    test_person_1.select(
        [
            "case_id",
            "num_group1",
            "incometype_1044T",
            "birth_259D",
            "empl_employedfrom_271D",
            "empl_industry_691L",
            "familystate_447L",
            "sex_738L",
            "type_25L",
            "safeguarantyflag_411L",
            "empl_employedtotal_800L",
            "role_1084L",
        ]
    )
    .filter(pl.col("num_group1") == 0)
    .drop("num_group1")
)

# we now have num_group1 and num_group2, so aggregate again
test_credit_bureau_b_2_feats = test_credit_bureau_b_2.group_by("case_id").agg(
    pl.col("pmts_pmtsoverdue_635A").sum().alias("pmts_pmtsoverdue_635A_sum"),
    pl.col("pmts_dpdvalue_108P").sum().alias("pmts_dpdvalue_108P_sum"),
)


# Additional aggregation for depth=1 files
test_other_1_feats = test_other_1.group_by("case_id").agg(
    pl.col("amtdebitincoming_4809443A").sum().alias("amtdebitincoming_4809443A_sum"),
    pl.col("amtdebitoutgoing_4809440A").sum().alias("amtdebitoutgoing_4809440A_sum"),
    pl.col("amtdepositbalance_4809441A").sum().alias("amtdepositbalance_4809441A_sum"),
    pl.col("amtdepositincoming_4809444A")
    .sum()
    .alias("amtdepositincoming_4809444A_sum"),
    pl.col("amtdepositoutgoing_4809442A")
    .sum()
    .alias("amtdepositoutgoing_4809442A_sum"),
)

test_credit_bureau_b_1_feats = test_credit_bureau_b_1.group_by("case_id").agg(
    pl.col("amount_1115A").sum().alias("amount_1115A_sum"),
    pl.col("credquantity_1099L").sum().alias("credquantity_1099L_sum"),
    pl.col("credquantity_984L").sum().alias("credquantity_984L_sum"),
    pl.col("debtpastduevalue_732A").sum().alias("debtpastduevalue_732A_sum"),
    pl.col("debtvalue_227A").sum().alias("debtvalue_227A_sum"),
    pl.col("dpd_550P").sum().alias("dpd_550P_sum"),
    pl.col("dpd_733P").sum().alias("dpd_733P_sum"),
    pl.col("dpdmax_851P").max().alias("dpdmax_851P_max"),
    pl.col("installmentamount_644A").sum().alias("installmentamount_644A_sum"),
    pl.col("installmentamount_833A").sum().alias("installmentamount_833A_sum"),
    pl.col("instlamount_892A").sum().alias("instlamount_892A_sum"),
    pl.col("interestrateyearly_538L").max().alias("interestrateyearly_538L_max"),
    pl.col("maxdebtpduevalodued_3940955A")
    .max()
    .alias("maxdebtpduevalodued_3940955A_max"),
    pl.col("numberofinstls_810L").sum().alias("numberofinstls_810L_sum"),
    pl.col("overdueamountmax_950A").max().alias("overdueamountmax_950A_max"),
    pl.col("pmtdaysoverdue_1135P").sum().alias("pmtdaysoverdue_1135P_sum"),
    pl.col("pmtnumpending_403L").sum().alias("pmtnumpending_403L_sum"),
    pl.col("residualamount_3940956A").sum().alias("residualamount_3940956A_sum"),
    pl.col("totalamount_503A").sum().alias("totalamount_503A_sum"),
    pl.col("totalamount_881A").sum().alias("totalamount_881A_sum"),
)

test_deposit_1_feats = test_deposit_1.group_by("case_id").agg(
    pl.col("amount_416A").sum().alias("amount_416A_sum")
)


join_data1 = (
    test_basetable.join(test_static, how="left", on="case_id")
    .join(test_static_cb, how="left", on="case_id")
    .join(test_person_1_feats_1, how="left", on="case_id")
    .join(test_person_1_feats_2, how="left", on="case_id")
    .join(test_credit_bureau_b_2_feats, how="left", on="case_id")
    .join(test_other_1_feats, how="left", on="case_id")
    .join(test_credit_bureau_b_1_feats, how="left", on="case_id")
    .join(test_deposit_1_feats, how="left", on="case_id")
)


join_data1 = join_data1.to_pandas()

join_data1.shape

del (
    test_static,
    test_static_cb,
    test_person_1,
    test_credit_bureau_b_2,
    test_other_1,
    test_credit_bureau_b_1,
    test_deposit_1,
)
del (
    test_person_1_feats_1,
    test_person_1_feats_2,
    test_credit_bureau_b_2_feats,
    test_other_1_feats,
    test_credit_bureau_b_1_feats,
    test_deposit_1_feats,
)
gc.collect()


# Additional testing data, depth = 1
test_applprev_1 = pl.concat(
    [
        pl.read_csv(pathway + "csv_files/test/test_applprev_1_0.csv").pipe(
            set_table_dtypes
        ),
        pl.read_csv(pathway + "csv_files/test/test_applprev_1_1.csv").pipe(
            set_table_dtypes
        ),
    ],
    how="vertical_relaxed",
)

test_tax_registry_a_1 = pl.read_csv(
    pathway + "csv_files/test/test_tax_registry_a_1.csv"
).pipe(set_table_dtypes)
test_tax_registry_b_1 = pl.read_csv(
    pathway + "csv_files/test/test_tax_registry_b_1.csv"
).pipe(set_table_dtypes)
test_tax_registry_c_1 = pl.read_csv(
    pathway + "csv_files/test/test_tax_registry_c_1.csv"
).pipe(set_table_dtypes)

test_credit_bureau_a_1 = pl.concat(
    [
        pl.read_csv(pathway + "csv_files/test/test_credit_bureau_a_1_0.csv").pipe(
            set_table_dtypes
        ),
        pl.read_csv(pathway + "csv_files/test/test_credit_bureau_a_1_1.csv").pipe(
            set_table_dtypes
        ),
        pl.read_csv(pathway + "csv_files/test/test_credit_bureau_a_1_2.csv").pipe(
            set_table_dtypes
        ),
        pl.read_csv(pathway + "csv_files/test/test_credit_bureau_a_1_3.csv").pipe(
            set_table_dtypes
        ),
    ],
    how="vertical_relaxed",
)

# Change L columns to float64
for col in test_credit_bureau_a_1.columns:
    if col[-1] in ("L"):
        test_credit_bureau_a_1 = test_credit_bureau_a_1.with_columns(
            pl.col(col).cast(pl.Float64).alias(col)
        )

test_applprev_1_feats_1 = test_applprev_1.group_by("case_id").agg(
    pl.col("actualdpd_943P").sum().alias("actualdpd_943P_sum"),
    pl.col("annuity_853A").sum().alias("annuity_853A_sum"),
    pl.col("byoccupationinc_3656910L").max().alias("byoccupationinc_3656910L_max"),
    pl.col("childnum_21L").max().alias("childnum_21L_max"),
    pl.col("credacc_credlmt_575A").max().alias("credacc_credlmt_575A_max"),
    pl.col("currdebt_94A").sum().alias("currdebt_94A_sum"),
    pl.col("downpmt_134A").sum().alias("downpmt_134A_sum"),
    pl.col("isbidproduct_390L").max(),
    pl.col("mainoccupationinc_437A").sum().alias("mainoccupationinc_437A_sum"),
    pl.col("maxdpdtolerance_577P").max().alias("maxdpdtolerance_577P_max"),
    pl.col("outstandingdebt_522A").sum().alias("outstandingdebt_522A_sum"),
    pl.col("pmtnum_8L").sum().alias("pmtnum_8L_sum"),
    pl.col("tenor_203L").sum().alias("tenor_203L_sum"),
)

test_applprev_1_feats_2 = (
    test_applprev_1.select(
        [
            "case_id",
            "num_group1",
            "credtype_587L",
            "familystate_726L",
            "inittransactioncode_279L",
            "status_219L",
        ]
    )
    .filter(pl.col("num_group1") == 0)
    .drop("num_group1")
)

test_tax_registry_a_1_feats = test_tax_registry_a_1.group_by("case_id").agg(
    pl.col("amount_4527230A").sum().alias("amount_4527230A_sum")
)

test_tax_registry_b_1_feats = test_tax_registry_b_1.group_by("case_id").agg(
    pl.col("amount_4917619A").sum().alias("amount_4917619A_sum")
)

test_tax_registry_c_1_feats = test_tax_registry_c_1.group_by("case_id").agg(
    pl.col("pmtamount_36A").sum().alias("pmtamount_36A_sum")
)

test_credit_bureau_a_1_feats = test_credit_bureau_a_1.group_by("case_id").agg(
    pl.col("dpdmax_139P").max().alias("dpdmax_139P_max"),
    pl.col("dpdmax_757P").max().alias("dpdmax_757P_max"),
    pl.col("monthlyinstlamount_332A").sum().alias("monthlyinstlamount_332A_sum"),
    pl.col("monthlyinstlamount_674A").sum().alias("monthlyinstlamount_674A_sum"),
    pl.col("nominalrate_498L").max().alias("nominalrate_498L_max"),
    pl.col("numberofinstls_229L").sum().alias("numberofinstls_229L_sum"),
    pl.col("numberofoutstandinstls_520L")
    .sum()
    .alias("numberofoutstandinstls_520L_sum"),
    pl.col("numberofoverdueinstlmax_1039L")
    .sum()
    .alias("numberofoverdueinstlmax_1039L_sum"),
    pl.col("numberofoverdueinstlmax_1151L")
    .sum()
    .alias("numberofoverdueinstlmax_1151L_sum"),
    pl.col("numberofoverdueinstls_725L").sum().alias("numberofoverdueinstls_725L_sum"),
    pl.col("numberofoverdueinstls_834L").sum().alias("numberofoverdueinstls_834L_sum"),
    pl.col("outstandingamount_354A").sum().alias("outstandingamount_354A_sum"),
    pl.col("overdueamount_31A").sum().alias("overdueamount_31A_sum"),
    pl.col("overdueamount_659A").sum().alias("overdueamount_659A_sum"),
    pl.col("overdueamountmax2_14A").max().alias("overdueamountmax2_14A_max"),
    pl.col("overdueamountmax2_398A").max().alias("overdueamountmax2_398A_max"),
    pl.col("overdueamountmax_155A").max().alias("overdueamountmax_155A_max"),
    pl.col("overdueamountmax_35A").max().alias("overdueamountmax_35A_max"),
    pl.col("periodicityofpmts_1102L").max().alias("periodicityofpmts_1102L_max"),
    pl.col("totalamount_6A").sum().alias("totalamount_6A_sum"),
)

test_tax_registry_c_1_feats = test_tax_registry_c_1_feats.with_columns(
    pl.col("case_id").cast(pl.Int64)
)

join_data2 = (
    test_basetable.join(test_applprev_1_feats_1, how="left", on="case_id")
    .join(test_applprev_1_feats_2, how="left", on="case_id")
    .join(test_tax_registry_a_1_feats, how="left", on="case_id")
    .join(test_tax_registry_b_1_feats, how="left", on="case_id")
    .join(test_tax_registry_c_1_feats, how="left", on="case_id")
    .join(test_credit_bureau_a_1_feats, how="left", on="case_id")
)

join_data2 = join_data2.to_pandas()

drop_cols = ["date_decision", "MONTH", "WEEK_NUM"]
join_data2 = join_data2.drop(drop_cols, axis=1, errors="ignore")
join_data2.shape


del (
    test_applprev_1,
    test_tax_registry_a_1,
    test_tax_registry_b_1,
    test_tax_registry_c_1,
    test_credit_bureau_a_1,
)
del (
    test_applprev_1_feats_1,
    test_applprev_1_feats_2,
    test_tax_registry_a_1_feats,
    test_tax_registry_b_1_feats,
    test_tax_registry_c_1_feats,
    test_credit_bureau_a_1_feats,
)
gc.collect()


# Additional testing data, depth = 2
test_applprev_2 = pl.read_csv(pathway + "csv_files/test/test_applprev_2.csv").pipe(
    set_table_dtypes
)

test_person_2 = pl.read_csv(pathway + "csv_files/test/test_person_2.csv").pipe(
    set_table_dtypes
)

sel = [
    "case_id",
    "num_group1",
    "num_group2",
    "pmts_dpd_1073P",
    "pmts_dpd_303P",
    "pmts_overdue_1140A",
    "pmts_overdue_1152A",
]
test_credit_bureau_a_2 = pl.concat(
    [
        pl.read_csv(
            pathway + "csv_files/test/test_credit_bureau_a_2_0.csv", columns=sel
        ).pipe(set_table_dtypes),
        pl.read_csv(
            pathway + "csv_files/test/test_credit_bureau_a_2_1.csv", columns=sel
        ).pipe(set_table_dtypes),
        pl.read_csv(
            pathway + "csv_files/test/test_credit_bureau_a_2_2.csv", columns=sel
        ).pipe(set_table_dtypes),
        pl.read_csv(
            pathway + "csv_files/test/test_credit_bureau_a_2_3.csv", columns=sel
        ).pipe(set_table_dtypes),
        pl.read_csv(
            pathway + "csv_files/test/test_credit_bureau_a_2_4.csv", columns=sel
        ).pipe(set_table_dtypes),
        pl.read_csv(
            pathway + "csv_files/test/test_credit_bureau_a_2_5.csv", columns=sel
        ).pipe(set_table_dtypes),
    ],
    how="vertical_relaxed",
)

test_credit_bureau_a_21 = pl.concat(
    [
        pl.read_csv(
            pathway + "csv_files/test/test_credit_bureau_a_2_6.csv", columns=sel
        ).pipe(set_table_dtypes),
        pl.read_csv(
            pathway + "csv_files/test/test_credit_bureau_a_2_7.csv", columns=sel
        ).pipe(set_table_dtypes),
        pl.read_csv(
            pathway + "csv_files/test/test_credit_bureau_a_2_8.csv", columns=sel
        ).pipe(set_table_dtypes),
        pl.read_csv(
            pathway + "csv_files/test/test_credit_bureau_a_2_9.csv", columns=sel
        ).pipe(set_table_dtypes),
        pl.read_csv(
            pathway + "csv_files/test/test_credit_bureau_a_2_10.csv", columns=sel
        ).pipe(set_table_dtypes),
    ],
    how="vertical_relaxed",
)

test_applprev_2_feats = (
    test_applprev_2.select(["case_id", "num_group1", "num_group2", "conts_type_509L"])
    .filter((pl.col("num_group1") == 0) & (pl.col("num_group2") == 0))
    .drop("num_group1")
    .drop("num_group2")
)

test_credit_bureau_a_2_feats = test_credit_bureau_a_2.group_by("case_id").agg(
    pl.col("pmts_dpd_1073P").sum().alias("pmts_dpd_1073P_sum"),
    pl.col("pmts_dpd_303P").sum().alias("pmts_dpd_303P_sum"),
    pl.col("pmts_overdue_1140A").sum().alias("pmts_overdue_1140A_sum"),
    pl.col("pmts_overdue_1152A").sum().alias("pmts_overdue_1152A_sum"),
)

test_credit_bureau_a_21_feats = test_credit_bureau_a_21.group_by("case_id").agg(
    pl.col("pmts_dpd_1073P").sum().alias("pmts_dpd_1073P_sum"),
    pl.col("pmts_dpd_303P").sum().alias("pmts_dpd_303P_sum"),
    pl.col("pmts_overdue_1140A").sum().alias("pmts_overdue_1140A_sum"),
    pl.col("pmts_overdue_1152A").sum().alias("pmts_overdue_1152A_sum"),
)

test_person_2_feats = (
    test_person_2.select(
        [
            "case_id",
            "num_group1",
            "num_group2",
            "addres_zip_823M",
            "addres_district_368M",
            "conts_role_79M",
            "empls_economicalst_849M",
            "empls_employer_name_740M",
        ]
    )
    .filter((pl.col("num_group1") == 0) & (pl.col("num_group2") == 0))
    .drop("num_group1")
    .drop("num_group2")
)

join_data3 = (
    test_basetable.join(test_applprev_2_feats, how="left", on="case_id")
    .join(test_credit_bureau_a_2_feats, how="left", on="case_id")
    .join(test_credit_bureau_a_21_feats, how="left", on="case_id")
    .join(test_person_2_feats, how="left", on="case_id")
)

join_data3 = join_data3.to_pandas()
join_data3 = join_data3.drop(
    ["date_decision", "MONTH", "WEEK_NUM"], axis=1, errors="ignore"
)

del test_applprev_2, test_person_2, test_credit_bureau_a_2, test_credit_bureau_a_21
del (
    test_applprev_2_feats,
    test_credit_bureau_a_2_feats,
    test_person_2_feats,
    test_credit_bureau_a_21_feats,
)
gc.collect()

join_data3.shape

dfs = [join_data1, join_data2, join_data3]
join_test = reduce(lambda left, right: pd.merge(left, right, on="case_id"), dfs)

join_test = pl.from_pandas(join_test)


join_test.head()


test = join_test.with_columns(pl.col("date_decision").cast(pl.Date))
# Feature engineer date columns
date_list = [
    "datefirstoffer_1144D",
    "datelastunpaid_3546854D",
    "dtlastpmtallstes_4499206D",
    "firstclxcampaign_1125D",
    "lastdelinqdate_224D",
    "lastrejectdate_50D",
    "maxdpdinstldate_3546855D",
    "birthdate_574D",
    "responsedate_1012D",
    "responsedate_4527233D",
    "responsedate_4917613D",
    "empl_employedfrom_271D",
    "firstdatedue_489D",
    "lastactivateddate_801D",
    "lastapplicationdate_877D",
    "lastapprdate_640D",
    "dateofbirth_337D",
    "birth_259D",
]

for col in date_list:
    test = test.with_columns(pl.col(col).cast(pl.Date))
    test = test.with_columns(
        ((pl.col("date_decision") - pl.col(col)) / (24 * 60 * 60 * 1000))
        .cast(pl.Float64)
        .alias(f"{col}_diff")
    )

test = test.pipe(set_table_dtypes).pipe(convert_strings)

drop_list = ["date_decision", "MONTH"] + date_list
# Convert to pandas for drop(errors='ignore')
test = test.to_pandas()
test = test.drop(drop_list, axis=1, errors="ignore")

del join_data1, join_data2, join_data3, join_test, dfs
gc.collect()

test.shape

test


# Noticed some of these numeric variables were parsed as strings, changed them all to int
num_list = [
    "numinstlswithdpd5_4187116L",
    "numinstmatpaidtearly2d_4499204L",
    "numinstpaid_4499208L",
    "numinstpaidearly3dest_4493216L",
    "numinstpaidearly5dest_4493211L",
    "numinstpaidearly5dobd_4499205L",
    "numinstpaidearlyest_4493214L",
    "numinstpaidlastcontr_4325080L",
    "numinstregularpaidest_4493210L",
    "numinsttopaygrest_4493213L",
    "numinstunpaidmaxest_4493212L",
    "contractssum_5085716L",
    "days120_123L",
    "days180_256L",
    "days360_512L",
    "firstquarter_103L",
    "fourthquarter_440L",
    "numberofqueries_373L",
    "pmtscount_423L",
    "secondquarter_766L",
    "thirdquarter_1082L",
    "days30_165L",
    "days90_310L",
]
for col in num_list:
    test[col] = test[col].astype("float64")

test

drop_list = [
    "lastapprcommoditytypec_5251766M",
    "lastrejectcommodtypec_5251769M",
    "lastrejectcommoditycat_161M",
    "lastrejectreasonclient_4145040M",
    "previouscontdistrict_112M",
    "lastapprcommoditycat_1041M",
    "lastcancelreason_561M",
    "lastrejectreason_759M",
]

test.drop(columns=drop_list, axis=1, errors="ignore", inplace=True)
test.dropna(axis=1, how="all", inplace=True)
test.head()


print(np.count_nonzero(test.isnull()))
# Impute missing values, 0 missing values after imputation
test = imputer(test)
print(np.count_nonzero(test.isnull()))

bool_cols = test.select_dtypes(include=["bool"]).columns.tolist()
for col in bool_cols:
    test[col] = test[col].astype(int)
for col in bool_cols:
    print(test[col].unique())


# save test data
test.to_csv("test_final.csv", index=False)
