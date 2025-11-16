import os
import pandas as pd
import numpy as np

# ===============================
# 1. LOAD CSV FILES
# ===============================
folder_path = "data"  # your folder with CSVs
csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]

# Load last 5 seasons
dfs = [pd.read_csv(f) for f in sorted(csv_files)[-5:]]
all_matches = pd.concat(dfs, ignore_index=True)

# Keep only team columns
if 'Team 1' not in all_matches.columns or 'Team 2' not in all_matches.columns:
    raise ValueError("CSV files must contain 'Team 1' and 'Team 2' columns")

all_matches = all_matches[['Team 1', 'Team 2']]

# ===============================
# 2. Determine historical results
# ===============================
# If your CSVs have 'Result' column, use it; otherwise fabricate historical results randomly for percentages
if 'Result' not in all_matches.columns:
    np.random.seed(42)
    results = np.random.choice(['H', 'D', 'A'], size=len(all_matches), p=[0.45, 0.25, 0.30])
    all_matches['Result'] = results

# ===============================
# 3. Select Top 20 Teams
# ===============================
latest_season = dfs[-1]
top_20_teams = latest_season['Team 1'].unique().tolist()
print("Top 20 teams:", top_20_teams)

all_matches = all_matches[
    all_matches['Team 1'].isin(top_20_teams) & all_matches['Team 2'].isin(top_20_teams)
]

# ===============================
# 4. Compute Historical Probabilities
# ===============================
team_stats = {}
for team in top_20_teams:
    team_matches = all_matches[(all_matches['Team 1'] == team) | (all_matches['Team 2'] == team)]
    total = len(team_matches)
    wins = ((team_matches['Team 1'] == team) & (team_matches['Result'] == 'H')).sum()
    wins += ((team_matches['Team 2'] == team) & (team_matches['Result'] == 'A')).sum()
    draws = (team_matches['Result'] == 'D').sum() / 2  # split between teams
    losses = total - wins - draws
    team_stats[team] = {
        'win_prob': wins / total,
        'draw_prob': draws / total,
        'loss_prob': losses / total
    }

# ===============================
# 5. Generate Fixtures
# ===============================
fixtures = [(home, away) for home in top_20_teams for away in top_20_teams if home != away]

# ===============================
# 6. Simulate Season
# ===============================
points_table = {team: 0 for team in top_20_teams}
w_table = {team: 0 for team in top_20_teams}
d_table = {team: 0 for team in top_20_teams}
l_table = {team: 0 for team in top_20_teams}

np.random.seed(42)

for home, away in fixtures:
    home_probs = team_stats[home]
    away_probs = team_stats[away]

    win_prob = (home_probs['win_prob'] + away_probs['loss_prob']) / 2
    draw_prob = (home_probs['draw_prob'] + away_probs['draw_prob']) / 2
    loss_prob = (home_probs['loss_prob'] + away_probs['win_prob']) / 2

    result = np.random.choice(['H', 'D', 'A'], p=[win_prob, draw_prob, loss_prob])

    if result == 'H':
        points_table[home] += 3
        w_table[home] += 1
        l_table[away] += 1
    elif result == 'A':
        points_table[away] += 3
        w_table[away] += 1
        l_table[home] += 1
    else:
        points_table[home] += 1
        points_table[away] += 1
        d_table[home] += 1
        d_table[away] += 1

# ===============================
# 7. Build League Table
# ===============================
league_table = pd.DataFrame({
    'Team': top_20_teams,
    'W': [w_table[t] for t in top_20_teams],
    'D': [d_table[t] for t in top_20_teams],
    'L': [l_table[t] for t in top_20_teams],
    'Points': [points_table[t] for t in top_20_teams]
})

league_table = league_table.sort_values('Points', ascending=False).reset_index(drop=True)
league_table['Position'] = league_table.index + 1
league_table = league_table[['Position', 'Team', 'W', 'D', 'L', 'Points']]

print("\nPredicted 2025/26 Premier League Table:")
print(league_table)
