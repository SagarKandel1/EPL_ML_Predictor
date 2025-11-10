# ===============================
# 1. IMPORTS
# ===============================

import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder




# ===============================
# 2. LOAD & MERGE CSV FILES
# ===============================


folder_path = "data"
print("Files found:", os.listdir(folder_path))

dataframe =[]
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        dataframe.append(df)

epl_data = pd.concat(dataframe, ignore_index=True)
print("data Loaded")
print("data shape:", epl_data.shape)
print("Columns:", epl_data.columns.tolist())

# ===============================
# 3. PARSE & PREPROCESS DATA
# ===============================
# Split 'FT' column into 'HomeGoals' and 'AwayGoals'
goals = epl_data['FT'].str.split('-', expand=True)
epl_data['FT'] = goals[0].astype(int)
epl_data['FA'] = goals[1].astype(int)

# Create 'Result' column
def determine_result(row):
    if row['FT'] > row['FA']:
        return 'H'
    elif row['FT'] < row['FA']:
        return 'A'
    else:
        return 'D'
epl_data['Result'] = epl_data.apply(determine_result, axis=1)

# ===============================
# 4. SELECT RELEVANT COLUMNS
# ===============================

df = epl_data[["Team 1", "Team 2","FT", "FA", "Result"]].copy()

le_home = LabelEncoder()
le_away = LabelEncoder()

df['Team 1'] = le_home.fit_transform(df['Team 1'])
df['Team 2'] = le_away.fit_transform(df['Team 2'])
print("Preprocessing completed.")

# ===============================
# 5. SPLIT DATA INTO TRAINING AND TEST SETS
# ===============================

X = df[['Team 1', 'Team 2', 'FT', 'FA']]
y = df['Result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets.")

# ===============================
# 6. STANDARDIZE FEATURES
# ===============================

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model training completed.")

# ===============================
# 7. EVALUATE MODEL
# ===============================

y_pred = model.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f"Model Evaluation completed.")
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# python
import numpy as np

# ===============================
# 8. PROMOTED TEAM STATS (fixed)
# ===============================

# Compute average stats for bottom 3 teams from the last seasons
season_stats = df.groupby('Team 1').agg({
    'FT': 'mean',
    'FA': 'mean',
    'Result': lambda x: (x == 'H').sum() * 3 + (x == 'D').sum()  # approximate points
}).reset_index()

bottom_teams = season_stats.nsmallest(3, 'Result')
avg_bottom_stats = bottom_teams[['FT', 'FA']].mean()

# Define promoted teams (names)
promoted_teams = ['Leeds United', 'Burnley', 'Sunderland']

# Ensure classes_ remain numpy arrays and extend with new teams
for team in promoted_teams:
    if team not in le_home.classes_:
        le_home.classes_ = np.concatenate([le_home.classes_, [team]])
    if team not in le_away.classes_:
        le_away.classes_ = np.concatenate([le_away.classes_, [team]])

# Get numeric ids for promoted teams
promoted_team_ids = le_home.transform(promoted_teams)

# ===============================
# 9. GENERATE ALL FIXTURES
# ===============================
existing_team_ids = df['Team 1'].unique().tolist()
teams_2025_26 = existing_team_ids + promoted_team_ids.tolist()

fixtures = []
for home in teams_2025_26:
    for away in teams_2025_26:
        if home != away:
            fixtures.append((home, away))

# ===============================
# 10. SIMULATE SEASON
# ===============================
promoted_set = set(promoted_team_ids.tolist())
points_table = {team: 0 for team in teams_2025_26}

for home, away in fixtures:
    input_row = pd.DataFrame({
        'Team 1': [home],
        'Team 2': [away],
        'FT': [int(avg_bottom_stats['FT']) if home in promoted_set else 0],
        'FA': [int(avg_bottom_stats['FA']) if away in promoted_set else 0]
    })
    pred = model.predict(input_row)[0]
    if pred == 'H':
        points_table[home] += 3
    elif pred == 'A':
        points_table[away] += 3
    else:
        points_table[home] += 1
        points_table[away] += 1

# ===============================
# 11. BUILD LEAGUE TABLE
# ===============================
league_table = pd.DataFrame({
    'team_id': list(points_table.keys()),
    'points': list(points_table.values())
}).sort_values('points', ascending=False).reset_index(drop=True)

# Map back to team names using inverse_transform
league_table['team'] = le_home.inverse_transform(league_table['team_id'].astype(int).tolist())
league_table['rank'] = league_table.index + 1
league_table = league_table[['rank', 'team', 'points']]

print("\nPredicted 2025/26 Premier League Table:")
print(league_table)
