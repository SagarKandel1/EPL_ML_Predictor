# =========================================
# EPL 2025/26 Prediction (Top 20 Teams Only)
# =========================================

import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import numpy as np

# ===============================
# 1. LOAD CSV FILES
# ===============================

folder_path = "data"  # folder where your CSVs are stored
dataframes = []

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(folder_path, filename))
        dataframes.append(df)

df_all = pd.concat(dataframes, ignore_index=True)
print(f"Loaded {len(dataframes)} CSV files, total matches: {df_all.shape[0]}")

# ===============================
# 2. DEFINE 2025/26 TEAMS (Top 20)
# ===============================

# Replace with the actual 17 staying + 3 promoted
top_20_teams = [
    'Arsenal', 'Man City', 'Man United', 'Chelsea', 'Liverpool', 'Tottenham',
    'Newcastle', 'Brighton', 'Aston Villa', 'West Ham', 'Brentford', 'Everton',
    'Fulham', 'Bournemouth', 'Crystal Palace', 'Nottingham Forest', 'Southampton',
    'Leeds United', 'Burnley', 'Sunderland'
]

# Filter historical matches to only include top 20 teams
df_all = df_all[df_all['Team 1'].isin(top_20_teams) & df_all['Team 2'].isin(top_20_teams)]

# ===============================
# 3. PARSE MATCH RESULTS (WIN/LOSS/DRAW)
# ===============================

df_all['Result'] = df_all['FT'].apply(lambda x: 'H' if int(x.split('-')[0]) > int(x.split('-')[1])
                                      else ('A' if int(x.split('-')[0]) < int(x.split('-')[1]) else 'D'))
# ===============================
# 4. ENCODE TEAM NAMES (All 20 teams)
# ===============================

# Fit on the top 20 teams, not just CSV teams
le_home = LabelEncoder()
le_away = LabelEncoder()

le_home.fit(top_20_teams)
le_away.fit(top_20_teams)

df_all['Team 1'] = le_home.transform(df_all['Team 1'])
df_all['Team 2'] = le_away.transform(df_all['Team 2'])

# ===============================
# 5. TRAIN RANDOM FOREST MODEL
# ===============================

X = df_all[['Team 1', 'Team 2']]
y = df_all['Result']

model = Pipeline([
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42))
])
model.fit(X, y)
print("RandomForest model trained on top 20 teams.")

# ===============================
# 6. GENERATE FIXTURES (20 TEAMS ONLY)
# ===============================

team_ids = le_home.transform(top_20_teams)
fixtures = [(home, away) for home in team_ids for away in team_ids if home != away]

# ===============================
# 7. SIMULATE SEASON & CALCULATE POINTS
# ===============================

points_table = {team: 0 for team in team_ids}

for home, away in fixtures:
    pred = model.predict(pd.DataFrame({'Team 1': [home], 'Team 2': [away]}))[0]
    if pred == 'H':
        points_table[home] += 3
    elif pred == 'A':
        points_table[away] += 3
    else:
        points_table[home] += 1
        points_table[away] += 1

# ===============================
# 8. BUILD LEAGUE TABLE
# ===============================

league_table = pd.DataFrame({
    'team_id': list(points_table.keys()),
    'points': list(points_table.values())
}).sort_values('points', ascending=False).reset_index(drop=True)

league_table['team'] = le_home.inverse_transform(league_table['team_id'].astype(int))
league_table['rank'] = league_table.index + 1
league_table = league_table[['rank', 'team', 'points']]

# ===============================
# 9. PRINT RESULTS
# ===============================

print("\nPredicted Premier League 2025/26 Table (Top 20 teams only):")
print(league_table)
