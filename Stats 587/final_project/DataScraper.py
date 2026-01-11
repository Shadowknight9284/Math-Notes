# Install latest nfl_data_py
import pandas as pd
import numpy as np
import nfl_data_py as nfl

print("Imports successful:", pd.__version__, np.__version__)

# Load 2024 play-by-play data
try:
    print("Loading 2024 play-by-play data...")
    pbp_2024 = nfl.import_pbp_data([2024])
    pbp = pbp_2024[pbp_2024['season_type'] == 'REG'].copy()
    print("Data loaded:", pbp.shape)
    print("Weeks included:", sorted(pbp['week'].unique()))
    print("Sample posteam values:", pbp['posteam'].head(10).tolist())
    if 'posteam' not in pbp.columns:
        raise KeyError("'posteam' missing from loaded data")
    print("Data shape (no NaN filter):", pbp.shape)
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# Pre-compute possession-based stats and quarter splits
try:
    print("Pre-computing possession stats and quarter splits...")
    pbp['home_poss'] = pbp['posteam'] == pbp['home_team']
    pbp['away_poss'] = pbp['posteam'] == pbp['away_team']
    pbp['first_half'] = pbp['qtr'].isin([1, 2])
    pbp['second_half'] = pbp['qtr'].isin([3, 4])
    pbp['overtime'] = pbp['qtr'] >= 5
    
    print("Aggregating 2024 data with standardized naming...")
    game_data = pbp.groupby(['season', 'week', 'game_id', 'home_team', 'away_team']).agg(
        point_diff=('total_home_score', 'last'),
        point_diff_away=('total_away_score', 'last'),
        off_yards_gained_h=('yards_gained', lambda x: x[pbp['home_poss']].sum()),
        off_yards_gained_a=('yards_gained', lambda x: x[pbp['away_poss']].sum()),
        off_pass_yards_h=('yards_gained', lambda x: x[pbp['home_poss'] & (pbp['pass'] == 1)].sum()),
        off_pass_yards_a=('yards_gained', lambda x: x[pbp['away_poss'] & (pbp['pass'] == 1)].sum()),
        off_rush_yards_h=('yards_gained', lambda x: x[pbp['home_poss'] & (pbp['rush'] == 1)].sum()),
        off_rush_yards_a=('yards_gained', lambda x: x[pbp['away_poss'] & (pbp['rush'] == 1)].sum()),
        off_turnovers_h=('fumble_lost', lambda x: (x[pbp['home_poss']].sum() + 
                                                  pbp['interception'][pbp['home_poss']].sum())),
        off_turnovers_a=('fumble_lost', lambda x: (x[pbp['away_poss']].sum() + 
                                                  pbp['interception'][pbp['away_poss']].sum())),
        off_touchdowns_h=('touchdown', lambda x: x[pbp['home_poss']].sum()),
        off_touchdowns_a=('touchdown', lambda x: x[pbp['away_poss']].sum()),
        off_epa_per_play_h=('epa', lambda x: x[pbp['home_poss']].mean()),
        off_epa_per_play_a=('epa', lambda x: x[pbp['away_poss']].mean()),
        def_epa_per_play_h=('epa', lambda x: x[pbp['defteam'] == pbp['home_team']].mean()),
        def_epa_per_play_a=('epa', lambda x: x[pbp['defteam'] == pbp['away_team']].mean()),
        off_wpa_h=('wpa', lambda x: x[pbp['home_poss']].sum()),
        off_wpa_a=('wpa', lambda x: x[pbp['away_poss']].sum()),
        def_wpa_h=('wpa', lambda x: x[pbp['defteam'] == pbp['home_team']].sum()),
        def_wpa_a=('wpa', lambda x: x[pbp['defteam'] == pbp['away_team']].sum()),
        off_cpoe_h=('cpoe', lambda x: x[pbp['home_poss'] & pbp['cpoe'].notna()].mean()),
        off_cpoe_a=('cpoe', lambda x: x[pbp['away_poss'] & pbp['cpoe'].notna()].mean()),
        off_success_rate_h=('success', lambda x: x[pbp['home_poss']].mean()),
        off_success_rate_a=('success', lambda x: x[pbp['away_poss']].mean()),
        off_qb_epa_h=('qb_epa', lambda x: x[pbp['home_poss']].sum()),
        off_qb_epa_a=('qb_epa', lambda x: x[pbp['away_poss']].sum()),
        def_sacks_h=('sack', lambda x: x[pbp['defteam'] == pbp['home_team']].sum()),
        def_sacks_a=('sack', lambda x: x[pbp['defteam'] == pbp['away_team']].sum()),
        def_qb_hits_h=('qb_hit', lambda x: x[pbp['defteam'] == pbp['home_team']].sum()),
        def_qb_hits_a=('qb_hit', lambda x: x[pbp['defteam'] == pbp['away_team']].sum()),
        off_red_zone_td_rate_h=('touchdown', lambda x: x[pbp['home_poss'] & (pbp['yardline_100'] <= 20)].sum() / 
                               max(1, (pbp['home_poss'] & (pbp['yardline_100'] <= 20)).sum())),
        off_red_zone_td_rate_a=('touchdown', lambda x: x[pbp['away_poss'] & (pbp['yardline_100'] <= 20)].sum() / 
                               max(1, (pbp['away_poss'] & (pbp['yardline_100'] <= 20)).sum())),
        off_third_down_conv_rate_h=('third_down_converted', lambda x: x[pbp['home_poss']].mean()),
        off_third_down_conv_rate_a=('third_down_converted', lambda x: x[pbp['away_poss']].mean()),
        off_fourth_down_conv_rate_h=('fourth_down_converted', lambda x: x[pbp['home_poss']].mean()),
        off_fourth_down_conv_rate_a=('fourth_down_converted', lambda x: x[pbp['away_poss']].mean()),
        off_first_downs_h=('first_down', lambda x: x[pbp['home_poss']].sum()),
        off_first_downs_a=('first_down', lambda x: x[pbp['away_poss']].sum()),
        off_pass_rate_h=('pass', lambda x: x[pbp['home_poss']].mean()),
        off_pass_rate_a=('pass', lambda x: x[pbp['away_poss']].mean()),
        off_shotgun_rate_h=('shotgun', lambda x: x[pbp['home_poss']].mean()),
        off_shotgun_rate_a=('shotgun', lambda x: x[pbp['away_poss']].mean()),
        off_no_huddle_rate_h=('no_huddle', lambda x: x[pbp['home_poss']].mean()),
        off_no_huddle_rate_a=('no_huddle', lambda x: x[pbp['away_poss']].mean()),
        timeouts_remaining_h=('home_timeouts_remaining', 'last'),
        timeouts_remaining_a=('away_timeouts_remaining', 'last'),
        score_differential_pre=('score_differential', 'mean'),
        total_plays=('play_id', 'count'),
        first_half_plays=('play_id', lambda x: x[pbp['first_half']].count()),
        second_half_plays=('play_id', lambda x: x[pbp['second_half']].count()),
        overtime_plays=('play_id', lambda x: x[pbp['overtime']].count()),
        roof=('roof', 'first'),
        surface=('surface', 'first')
    ).reset_index()

    # Compute outcome flags
    game_data['point_diff'] = game_data['point_diff'] - game_data['point_diff_away']
    game_data['home_win'] = (game_data['point_diff'] > 0).astype(int)
    game_data['away_wins'] = (game_data['point_diff'] < 0).astype(int)
    game_data['tie_flag'] = (game_data['point_diff'] == 0).astype(int)
    game_data.drop(columns=['point_diff_away'], inplace=True)
except Exception as e:
    print(f"Error aggregating data: {e}")
    raise

# Load 2014-2024 H2H and team performance data
print("Loading 2014-2024 data for H2H and team stats...")
pbp_historical = nfl.import_pbp_data(range(2014, 2025))
pbp_historical = pbp_historical[pbp_historical['season_type'] == 'REG'].copy()
historical_games = pbp_historical.groupby(['season', 'week', 'game_id', 'home_team', 'away_team']).agg(
    home_score=('total_home_score', 'last'),
    away_score=('total_away_score', 'last')
).reset_index()
historical_games['home_team'] = historical_games['home_team'].replace('STL', 'LA')
historical_games['away_team'] = historical_games['away_team'].replace('STL', 'LA')
game_data['home_team'] = game_data['home_team'].replace('STL', 'LA')
game_data['away_team'] = game_data['away_team'].replace('STL', 'LA')
historical_games['winner'] = np.where(historical_games['home_score'] > historical_games['away_score'], 
                                      historical_games['home_team'],
                                      np.where(historical_games['home_score'] < historical_games['away_score'], 
                                               historical_games['away_team'], None))

# Compute H2H (10-season with ties)
print("Computing 10-season H2H...")
def compute_h2h_prior(row, hist_games):
    prior_games = hist_games[
        ((hist_games['season'] < 2024) | 
         ((hist_games['season'] == 2024) & (hist_games['week'] < row['week']))) &
        (((hist_games['home_team'] == row['home_team']) & (hist_games['away_team'] == row['away_team'])) |
         ((hist_games['home_team'] == row['away_team']) & (hist_games['away_team'] == row['home_team'])))
    ]
    home_team_wins = prior_games[prior_games['winner'] == row['home_team']].shape[0]
    away_team_wins = prior_games[prior_games['winner'] == row['away_team']].shape[0]
    ties = prior_games[prior_games['winner'].isna()].shape[0]
    return pd.Series({
        'home_team_wins_10season': home_team_wins,
        'away_team_wins_10season': away_team_wins,
        'ties_10season': ties
    })

game_data[['home_team_wins_10season', 'away_team_wins_10season', 'ties_10season']] = game_data.apply(
    lambda row: compute_h2h_prior(row, historical_games), axis=1
)

# Compute win/loss/tie streaks, points last 3 games, and season record to date
print("Computing team performance metrics and season records...")
def compute_team_stats(row, hist_games, team_type):
    team = row[team_type]
    current_week = row['week']
    # Prior games for streaks and points (last 3)
    prior_games_last_3 = hist_games[
        ((hist_games['season'] < 2024) | 
         ((hist_games['season'] == 2024) & (hist_games['week'] < current_week))) &
        ((hist_games['home_team'] == team) | (hist_games['away_team'] == team))
    ].sort_values(['season', 'week'], ascending=False).head(3)
    wins_last_3 = prior_games_last_3[prior_games_last_3['winner'] == team].shape[0]
    losses_last_3 = prior_games_last_3[prior_games_last_3['winner'] != team].shape[0]
    points_scored = prior_games_last_3.apply(
        lambda x: x['home_score'] if x['home_team'] == team else x['away_score'], axis=1
    ).mean()
    points_allowed = prior_games_last_3.apply(
        lambda x: x['away_score'] if x['home_team'] == team else x['home_score'], axis=1
    ).mean()
    
    # Season record to date (all prior 2024 games)
    prior_games_season = hist_games[
        (hist_games['season'] == 2024) & 
        (hist_games['week'] < current_week) &
        ((hist_games['home_team'] == team) | (hist_games['away_team'] == team))
    ]
    wins_to_date = prior_games_season[prior_games_season['winner'] == team].shape[0]
    losses_to_date = prior_games_season[
        ((prior_games_season['home_team'] == team) & (prior_games_season['home_score'] < prior_games_season['away_score'])) |
        ((prior_games_season['away_team'] == team) & (prior_games_season['away_score'] < prior_games_season['home_score']))
    ].shape[0]
    ties_to_date = prior_games_season[prior_games_season['home_score'] == prior_games_season['away_score']].shape[0]
    
    return pd.Series({
        f'{team_type}_win_streak_last_3': wins_last_3,
        f'{team_type}_losing_streak_last_3': losses_last_3,
        f'{team_type}_points_scored_last_3': points_scored,
        f'{team_type}_points_allowed_last_3': points_allowed,
        f'{team_type}_wins_to_date': wins_to_date,
        f'{team_type}_losses_to_date': losses_to_date,
        f'{team_type}_ties_to_date': ties_to_date
    })

game_data[['home_win_streak_last_3', 'home_losing_streak_last_3', 'home_points_scored_last_3', 
           'home_points_allowed_last_3', 'home_wins_to_date', 'home_losses_to_date', 'home_ties_to_date']] = game_data.apply(
    lambda row: compute_team_stats(row, historical_games, 'home_team'), axis=1
)
game_data[['away_win_streak_last_3', 'away_losing_streak_last_3', 'away_points_scored_last_3', 
           'away_points_allowed_last_3', 'away_wins_to_date', 'away_losses_to_date', 'away_ties_to_date']] = game_data.apply(
    lambda row: compute_team_stats(row, historical_games, 'away_team'), axis=1
)

# Conference and division mapping
team_divisions = {
    'AFC East': ['BUF', 'MIA', 'NE', 'NYJ'], 'AFC North': ['BAL', 'CIN', 'CLE', 'PIT'],
    'AFC South': ['HOU', 'IND', 'JAX', 'TEN'], 'AFC West': ['DEN', 'KC', 'LAC', 'LV'],
    'NFC East': ['DAL', 'NYG', 'PHI', 'WAS'], 'NFC North': ['CHI', 'DET', 'GB', 'MIN'],
    'NFC South': ['ATL', 'CAR', 'NO', 'TB'], 'NFC West': ['ARI', 'LA', 'SF', 'SEA']
}
team_to_division = {team: div for div, teams in team_divisions.items() for team in teams}
team_to_conference = {team: div.split()[0] for team, div in team_to_division.items()}

print("Adding conference and division variables...")
game_data['same_conference'] = game_data.apply(
    lambda row: team_to_conference[row['home_team']] == team_to_conference[row['away_team']], axis=1
)
game_data['same_division'] = game_data.apply(
    lambda row: team_to_division[row['home_team']] == team_to_division[row['away_team']], axis=1
)

# Special games and additional schedule data
print("Loading 2024 schedule for special games and additional data...")
schedule_2024 = nfl.import_schedules([2024])
special_game_ids = schedule_2024[
    ((schedule_2024['week'] == 13) & (
        ((schedule_2024['home_team'] == 'DET') & (schedule_2024['away_team'] == 'CHI')) |
        ((schedule_2024['home_team'] == 'DAL') & (schedule_2024['away_team'] == 'NYG')) |
        ((schedule_2024['home_team'] == 'GB') & (schedule_2024['away_team'] == 'MIA'))
    )) | 
    ((schedule_2024['week'] == 13) & (schedule_2024['home_team'] == 'KC') & (schedule_2024['away_team'] == 'LV')) |
    ((schedule_2024['week'] == 17) & (
        ((schedule_2024['home_team'] == 'PIT') & (schedule_2024['away_team'] == 'KC')) |
        ((schedule_2024['home_team'] == 'HOU') & (schedule_2024['away_team'] == 'BAL'))
    ))
]['game_id'].tolist()
game_data['special_game'] = game_data['game_id'].isin(special_game_ids)

# Merge schedule data
schedule_subset = schedule_2024[['game_id', 'gameday', 'gametime', 'spread_line', 'total_line']]
game_data = game_data.merge(schedule_subset, on='game_id', how='left')
game_data['outdoor_game'] = game_data['roof'].apply(lambda x: x in ['outdoors', 'open'] if pd.notna(x) else False)

# Clean NaNs with updated variable names
game_data.fillna({
    'off_cpoe_h': 0, 'off_cpoe_a': 0, 
    'off_red_zone_td_rate_h': 0, 'off_red_zone_td_rate_a': 0,
    'off_third_down_conv_rate_h': 0, 'off_third_down_conv_rate_a': 0,
    'off_fourth_down_conv_rate_h': 0, 'off_fourth_down_conv_rate_a': 0,
    'home_team_wins_10season': 0, 'away_team_wins_10season': 0, 'ties_10season': 0,
    'home_win_streak_last_3': 0, 'away_win_streak_last_3': 0,
    'home_losing_streak_last_3': 0, 'away_losing_streak_last_3': 0,
    'home_points_scored_last_3': 0, 'away_points_scored_last_3': 0,
    'home_points_allowed_last_3': 0, 'away_points_allowed_last_3': 0,
    'home_wins_to_date': 0, 'away_wins_to_date': 0,
    'home_losses_to_date': 0, 'away_losses_to_date': 0,
    'home_ties_to_date': 0, 'away_ties_to_date': 0,
    'spread_line': 0, 'total_line': 0,
    'total_plays': 0, 'first_half_plays': 0, 'second_half_plays': 0, 'overtime_plays': 0
}, inplace=True)

# Round all numeric columns to 3 decimal places
numeric_cols = game_data.select_dtypes(include=[np.number]).columns
game_data[numeric_cols] = game_data[numeric_cols].round(3)

# Save and inspect
game_data.to_csv('NFL_DataScrap(2024-2025).csv', index=False)
print("Data saved to 'NFL_DataScrap(2024-2025).csv'")
print(game_data[['game_id', 'week', 'home_team', 'away_team', 'home_win', 'away_wins', 'tie_flag',
                  'off_wpa_h', 'off_wpa_a', 'def_wpa_h', 'def_wpa_a', 'off_epa_per_play_h', 
                  'off_epa_per_play_a', 'def_epa_per_play_h', 'def_epa_per_play_a',
                  'home_team_wins_10season', 'away_team_wins_10season', 'ties_10season',
                  'same_conference', 'same_division', 'special_game', 'home_win_streak_last_3', 
                  'away_win_streak_last_3', 'home_losing_streak_last_3', 'away_losing_streak_last_3', 
                  'home_points_scored_last_3', 'away_points_scored_last_3', 'home_wins_to_date', 
                  'home_losses_to_date', 'home_ties_to_date', 'away_wins_to_date', 'away_losses_to_date', 
                  'away_ties_to_date', 'total_plays', 'first_half_plays', 'second_half_plays', 
                  'overtime_plays', 'gameday', 'gametime', 'spread_line', 'total_line', 'outdoor_game']].head(10))
print("Special games count:", game_data['special_game'].sum())
print("Max home_team_wins_10season:", game_data['home_team_wins_10season'].max())
print("Max away_team_wins_10season:", game_data['away_team_wins_10season'].max())
print("LA vs. ARI games:")
print(game_data[(game_data['home_team'].isin(['LA', 'ARI'])) & (game_data['away_team'].isin(['LA', 'ARI']))][
    ['game_id', 'week', 'home_team', 'away_team', 'home_win', 'away_wins', 'tie_flag',
     'off_wpa_h', 'off_wpa_a', 'def_wpa_h', 'def_wpa_a', 'off_epa_per_play_h', 
     'off_epa_per_play_a', 'def_epa_per_play_h', 'def_epa_per_play_a', 
     'same_conference', 'same_division', 'special_game', 'home_win_streak_last_3', 
     'away_win_streak_last_3', 'home_losing_streak_last_3', 'away_losing_streak_last_3', 
     'home_points_scored_last_3', 'away_points_scored_last_3', 'home_wins_to_date', 
     'home_losses_to_date', 'home_ties_to_date', 'away_wins_to_date', 'away_losses_to_date', 
     'away_ties_to_date', 'total_plays', 'first_half_plays', 'second_half_plays', 
     'overtime_plays', 'gameday', 'gametime', 'spread_line', 'total_line', 'outdoor_game']])
print("Columns:", game_data.columns.tolist())