# ===============================
# 1. IMPORTS
# ===============================
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ===============================
# 2. LOAD CSV FILES
# ===============================
folder_path = "data"
csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

all_matches = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
print(f"Loaded {len(csv_files)} CSV files, total matches: {len(all_matches)}")

# ===============================
# 3. DEFINE TOP 20 TEAMS
# ===============================
top_20_teams = [
    'Arsenal', 'Crystal Palace', 'West Ham', 'Tottenham', 'Newcastle',
    'Man United', 'Man City', 'Everton', 'Liverpool', 'Chelsea',
    'Brighton', 'Wolves', 'Southampton', 'Leicester', 'Aston Villa',
    'Burnley', 'Bournemouth', 'Fulham', 'Leeds', 'Brentford'
]

# Filter matches: only keep those between top 20 teams
all_matches = all_matches[
    all_matches['Team 1'].isin(top_20_teams) &
    all_matches['Team 2'].isin(top_20_teams)
    ].copy()

# Split 'FT' into Home and Away goals if exists
if 'FT' in all_matches.columns:
    goals = all_matches['FT'].str.split('-', expand=True)
    all_matches['FT'] = goals[0].astype(int)
    all_matches['FA'] = goals[1].astype(int)
else:
    all_matches['FT'] = 0
    all_matches['FA'] = 0


# ===============================
# 4. DETERMINE MATCH RESULTS
# ===============================
def get_result(row):
    if row['FT'] > row['FA']:
        return 'H'
    elif row['FT'] < row['FA']:
        return 'A'
    else:
        return 'D'


all_matches['Result'] = all_matches.apply(get_result, axis=1)

# ===============================
# 5. PREPARE DATA FOR ML
# ===============================
le_home = LabelEncoder()
le_away = LabelEncoder()
le_home.fit(top_20_teams)
le_away.fit(top_20_teams)

all_matches['Team 1'] = le_home.transform(all_matches['Team 1'])
all_matches['Team 2'] = le_away.transform(all_matches['Team 2'])

X = all_matches[['Team 1', 'Team 2']]
y = all_matches['Result']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===============================
# 6. TRAIN RANDOM FOREST MODEL
# ===============================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = (model.predict(X_test) == y_test).mean()
print(f"Model trained. Test accuracy: {accuracy * 100:.2f}%")

# ===============================
# 7. GENERATE FIXTURES
# ===============================
team_ids = le_home.transform(top_20_teams)
fixtures = []

# Each team plays each other twice (home and away)
for home in team_ids:
    for away in team_ids:
        if home != away:
            fixtures.append((home, away))

# ===============================
# 8. SIMULATE SEASON
# ===============================
np.random.seed(None)  # Truly random each run

points_table = {team: {'W': 0, 'D': 0, 'L': 0, 'Points': 0} for team in team_ids}

for home, away in fixtures:
    input_row = pd.DataFrame({'Team 1': [home], 'Team 2': [away]})

    # Predict result probabilistically using model
    pred = model.predict(input_row)[0]

    if pred == 'H':
        points_table[home]['W'] += 1
        points_table[home]['Points'] += 3
        points_table[away]['L'] += 1
    elif pred == 'A':
        points_table[away]['W'] += 1
        points_table[away]['Points'] += 3
        points_table[home]['L'] += 1
    else:  # Draw
        points_table[home]['D'] += 1
        points_table[home]['Points'] += 1
        points_table[away]['D'] += 1
        points_table[away]['Points'] += 1

# ===============================
# 9. BUILD FINAL LEAGUE TABLE
# ===============================
league_table = pd.DataFrame([
    {
        'Team': le_home.inverse_transform([team])[0],
        'W': stats['W'],
        'D': stats['D'],
        'L': stats['L'],
        'Points': stats['Points']
    }
    for team, stats in points_table.items()
])

league_table = league_table.sort_values('Points', ascending=False).reset_index(drop=True)
league_table['Position'] = league_table.index + 1
league_table = league_table[['Position', 'Team', 'W', 'D', 'L', 'Points']]

print("\nPredicted 2025/26 Premier League Table:")
print(league_table)
