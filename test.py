import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

# # Load your data
df = pd.read_csv("TSdata/training.csv")
# Filter rows where series_1 == 0
series_1_data = df[(df['serieNames'] == 'serie_1') & (df['sales'] == 0)]
total_series_1_zero = len(series_1_data)

# Filter data where serieName is 'series_2' and value is 5, but only from series_1 == 0 rows
series_2_data = df[(df['serieNames'] == 'serie_2') & (df['sales'] == 5)]

matched_dates = series_1_data['TSDate']
series_2_with_series_1_zero = series_2_data[series_2_data['TSDate'].isin(matched_dates)].shape[0]

# Total occurrences of series_1 == 0
total_series_1_zero = series_1_data.shape[0]

# Calculate percentage
percentage = (series_2_with_series_1_zero / total_series_1_zero) * 100 if total_series_1_zero > 0 else 0

# Print results
print(f"Total occurrences of series_1 == 0: {total_series_1_zero}")
print(f"Occurrences of series_2 == 5 matching TSDate of series_1 == 0: {series_2_with_series_1_zero}")
print(f"Percentage of series_2 == 5 within series_1 == 0 (matched by TSDate): {percentage:.2f}%")

df2 = pd.read_csv("TSdata/test.csv")
# Ensure the date column is in datetime format
df["TSDate"] = pd.to_datetime(df["TSDate"])
df2["TSDate"] = pd.to_datetime(df2["TSDate"])

df = pd.concat([df, df2]).reset_index(drop=True)

# Generate basic date features
df["year"] = df["TSDate"].dt.year
df["month"] = df["TSDate"].dt.month
df["day"] = df["TSDate"].dt.day
df["weekday"] = df["TSDate"].dt.day_name()
df["weekday_num"] = df["TSDate"].dt.weekday  # Monday=0, Sunday=6
df["week_number"] = df["TSDate"].dt.isocalendar().week
df["is_weekend"] = df["weekday_num"] >= 5



# Function to calculate Easter Sunday based on the "Computus" algorithm
def get_easter(year):
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return pd.Timestamp(year, month, day)

# Define custom calendar events
# Define a mapping for events to integers
event_mapping = {
    "Liberation Day": 1,
    "Mother's Day": 2,
    "Father's Day": 3,
    "Veterans Day": 4,
    "Valentine's Day": 5,
    "Halloween": 6,
    "Thanksgiving": 7,
    "Black Friday": 8,
    "Cyber Monday": 9,
    "Easter": 10,
    "Easter Monday": 11,
    "Sinterklaas": 12,
    "Christmas": 13,  # Both 1st and 2nd Christmas Day will share the same ID
    None: 0  # No event
}

def dutch_calendar_events(date):
    year = date.year
    easter = get_easter(year)
    mothers_day = pd.Timestamp(year, 5, 1) + timedelta(days=(6 - pd.Timestamp(year, 5, 1).weekday() + 7))  # 2nd Sunday of May
    fathers_day = pd.Timestamp(year, 6, 1) + timedelta(days=(6 - pd.Timestamp(year, 6, 1).weekday() + 14))  # 3rd Sunday of June
    veterans_day = pd.Timestamp(year, 6, 1) + timedelta(days=(5 - pd.Timestamp(year, 6, 1).weekday() + 27))  # Last Saturday of June
    events = {
        "Liberation Day": date.month == 5 and date.day == 5,
        "Mother's Day": date == mothers_day,
        "Father's Day": date == fathers_day,
        "Veterans Day": date == veterans_day,
        "Valentine's Day": date.month == 2 and date.day == 14,
        "Halloween": date.month == 10 and date.day == 31,
        "Thanksgiving": date.month == 11 and date.weekday() == 3 and 22 <= date.day <= 28,
        "Black Friday": date.month == 11 and date.weekday() == 4 and 23 <= date.day <= 29,
        "Cyber Monday": date.month == 11 and date.weekday() == 0 and 26 <= date.day <= 30,
        "Easter": date == easter,
        "Easter Monday": date == (easter + timedelta(days=1)),
        "Sinterklaas": date.month == 12 and date.day == 5,
        "Christmas": date.month == 12 and date.day in [25, 26],  # Includes 1st and 2nd Christmas Day
    }
    for event, condition in events.items():
        if condition:
            return event
    return None

# Apply Dutch calendar events and map them to integers
df["dutch_event"] = df["TSDate"].apply(dutch_calendar_events)
df["dutch_event"] = df["dutch_event"].map(event_mapping).astype(int)

# Create boolean columns for each event
events = [
    "Liberation Day", "Mother's Day", "Father's Day", "Veterans Day", "Valentine's Day",
    "Halloween", "Thanksgiving", "Black Friday", "Cyber Monday", "Easter", "Easter Monday",
    "Sinterklaas", "Christmas"
]
for event in events:
    column_name = f"""is_{event.lower().replace(' ', '_').replace("'", '')}"""
    df[column_name] = df["dutch_event"] == event



# Add Season Number
def get_season(month):
    if month in [12, 1, 2]:
        return 1  # Winter
    elif month in [3, 4, 5]:
        return 2  # Spring
    elif month in [6, 7, 8]:
        return 3  # Summer
    elif month in [9, 10, 11]:
        return 4  # Autumn

df["season"] = df["month"].apply(get_season)

# Sort by series and date to ensure proper ordering
df = df.sort_values(by=[ "TSDate"]).reset_index(drop=True)

# Add Lagged Features (1 to 7 days)
for lag in range(1, 8):
    df[f"lag_{lag}"] = df.groupby("serieNames")["sales"].shift(lag)

# Add Rolling Statistics (last 7 days)
df["rolling_min"] = df.groupby("serieNames")["sales"].transform(lambda x: x.rolling(window=7).min())
df["rolling_max"] = df.groupby("serieNames")["sales"].transform(lambda x: x.rolling(window=7).max())
df["rolling_std"] = df.groupby("serieNames")["sales"].transform(lambda x: x.rolling(window=7).std())

df.drop(columns=["weekday"], inplace=True)
boolean_columns = df.select_dtypes(include="bool").columns
df[boolean_columns] = df[boolean_columns].astype(int)
# Save to CSV
df.to_csv("ts_features_dutch_calendar_dataset.csv", index=False)

# Display some plots
# 1. Sales over time
# plt.figure(figsize=(10, 6))
# plt.plot(df["TSDate"], df["sales"], label="Sales")
# plt.title("Sales Over Time")
# plt.xlabel("Date")
# plt.ylabel("Sales")
# plt.legend()
# plt.show()

# # 2. Average sales by weekday
# weekday_sales = df.groupby("weekday")["sales"].mean().reindex(
#     ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
# )
# plt.figure(figsize=(8, 5))
# weekday_sales.plot(kind="bar")
# plt.title("Average Sales by Weekday")
# plt.xlabel("Weekday")
# plt.ylabel("Average Sales")
# plt.show()

# # 3. Sales distribution by custom Dutch events
# event_sales = df.groupby("dutch_event")["sales"].mean()
# plt.figure(figsize=(10, 6))
# event_sales.plot(kind="bar")
# plt.title("Average Sales by Dutch Calendar Events")
# plt.xlabel("Dutch Event")
# plt.ylabel("Average Sales")
# plt.xticks(rotation=45)
# plt.show()


# import matplotlib.pyplot as plt


# Filter rows with zero sales
# zero_sales = df[df["sales"] ==  0.0]

# print(zero_sales.head())   

# # Count occurrences of zero sales by date
# zero_sales_by_date = zero_sales.groupby("TSDate").size()

# #Plot zero sales by date
# plt.figure(figsize=(10, 6))
# plt.scatter(zero_sales_by_date.index, zero_sales_by_date.values, color='red', label="Zero Sales")
# plt.title("Zero Sales Over Time")
# plt.xlabel("Date")
# plt.ylabel("Number of Zero Sales")
# plt.legend()
# plt.show()

# # Count occurrences of zero sales by weekday
# zero_sales_by_weekday = zero_sales.groupby("weekday").size().reindex(
#     ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
# )

# # Plot zero sales by weekday
# plt.figure(figsize=(8, 5))
# zero_sales_by_weekday.plot(kind="bar", color='orange')
# plt.title("Zero Sales by Weekday")
# plt.xlabel("Weekday")
# plt.ylabel("Number of Zero Sales")
# plt.show()

# # Optional: Print dates with zero sales for inspection
# print("Dates with Zero Sales:")
# print(zero_sales["TSDate"].unique())
