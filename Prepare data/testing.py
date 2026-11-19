import pandas as pd

# Load metadata
df = pd.read_csv("Prepare data/SLEEP_STUDY.csv")

# Keep only one sleep study per patient
df = (
    df.sort_values("SLEEP_STUDY_START_DATETIME")
      .drop_duplicates(subset="STUDY_PAT_ID", keep="first")
)

# Your age group definition
def age_group_for_sleep(days):
    if days < 2 * 365:
        return 3    # infant
    elif days < 12 * 365:
        return 2    # child
    elif days < 18 * 365:
        return 1    # teen
    else:
        return 0    # adult

df["AGE_GROUP"] = df["AGE_AT_SLEEP_STUDY_DAYS"].apply(age_group_for_sleep)

# Exclude teens
df = df[df["AGE_GROUP"] != 1]

# Sample 100 from each group
N_PER_GROUP = 100

infants = df[df["AGE_GROUP"] == 3].sample(n=N_PER_GROUP, random_state=42)
children = df[df["AGE_GROUP"] == 2].sample(n=N_PER_GROUP, random_state=43)
adults = df[df["AGE_GROUP"] == 0].sample(n=N_PER_GROUP, random_state=44)

sample = pd.concat([infants, children, adults]).sample(frac=1, random_state=45)

# Build download links
BASE = "https://physionet.org/files/nch-sleep/3.1.0/Sleep_Data/"

sample["EDF_LINK"] = (
    BASE
    + sample["STUDY_PAT_ID"].astype(str)
    + '_'
    + sample["SLEEP_STUDY_ID"].astype(str)
    + ".edf?download"
)

sample["TSV_LINK"] = (
    BASE
    + sample["STUDY_PAT_ID"].astype(str)
    + '_'
    + sample["SLEEP_STUDY_ID"].astype(str)
    + ".tsv?download"
)

# Save results
r = sample[[
    "STUDY_PAT_ID",
    "SLEEP_STUDY_ID",
    "AGE_GROUP",
    "EDF_LINK",
    "TSV_LINK"
]]

for row in r.itertuples(index=False):
    print(f"{row.EDF_LINK}")
    print(f"{row.TSV_LINK}")