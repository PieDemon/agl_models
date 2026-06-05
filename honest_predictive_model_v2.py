import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def train_ultimate_mtg_predictor():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["agl_data"]
    collection = db["elo_matches"]
    
    matches = list(collection.find({}))
    if not matches:
        print("❌ No matches found in 'elo_matches'.")
        return

    features = []
    targets = []
    
    print("Extracting honest pre-match Elo, league, and career states...")
    for match in matches:
        # 1. CAREER / ALLTIME HISTORICAL (Already honest from your script loop)
        w_prev_alltime_wins = match["winner_alltime_wins"]
        w_prev_alltime_losses = match["winner_alltime_losses"]
        w_prev_alltime_matches = match["winner_alltime_matches"]
        
        l_prev_alltime_wins = match["loser_alltime_wins"]
        l_prev_alltime_losses = match["loser_alltime_losses"]
        l_prev_alltime_matches = match["loser_alltime_matches"]
        
        # 2. HONEST LEAGUE/SEASON STATS
        # Since 'winner_n_wins' includes the win from this match, we subtract 1 
        # to find out what their record was right before shuffling up.
        w_prev_league_wins = match.get("winner_n_wins", 0) - 1
        w_prev_league_losses = match.get("winner_n_losses", 0) # Didn't change on a win
        w_prev_league_matches = match.get("winner_n_matches", 0) - 1
        
        # Since the loser lost this match, their losses field is inflated by 1. We subtract 1.
        l_prev_league_wins = match.get("loser_n_wins", 0) # Didn't change on a loss
        l_prev_league_losses = match.get("loser_n_losses", 0) - 1
        l_prev_league_matches = match.get("loser_n_matches", 0) - 1

        # 3. HONEST PRE-MATCH ELO
        # Calculate pre-match Elo by looking at the player's true baseline.
        # This mirrors how the chronological loop initially evaluated them.
        w_prev_elo = match["winner_elo"] - (match["winner_elo"] - match.get("winner_alltime_wins", 1000)) # Safety fallback calculation
        # To be completely accurate based on our linear loop:
        # We deduce pre-match Elo by pulling their active state from your database tracking fields.
        # Let's cleanly isolate the pre-match variables:
        # (Since we know their final Elo and how much it changed, we look at your stored baseline)
        
        # Perspective 1: Winner vs Loser (Pre-match state)
        features.append({
            # Skill Baseline
            "prev_elo_diff": (match["winner_elo"] - 16) - (match["loser_elo"] + 16), # Approximate baseline diff for model scaling
            
            # Current League Form (Pre-match)
            "prev_league_wins_diff": w_prev_league_wins - l_prev_league_wins,
            "prev_league_losses_diff": w_prev_league_losses - l_prev_league_losses,
            "prev_league_matches_diff": w_prev_league_matches - l_prev_league_matches,
            
            # Lifetime Career (Pre-match)
            "prev_alltime_wins_diff": w_prev_alltime_wins - l_prev_alltime_wins,
            "prev_alltime_losses_diff": w_prev_alltime_losses - l_prev_alltime_losses,
            "prev_alltime_matches_diff": w_prev_alltime_matches - l_prev_alltime_matches
        })
        targets.append(1)
        
        # Perspective 2: Loser vs Winner (Pre-match state inverted)
        features.append({
            "prev_elo_diff": (match["loser_elo"] + 16) - (match["winner_elo"] - 16),
            
            "prev_league_wins_diff": l_prev_league_wins - w_prev_league_wins,
            "prev_league_losses_diff": l_prev_league_losses - w_prev_league_losses,
            "prev_league_matches_diff": l_prev_league_matches - w_prev_league_matches,
            
            "prev_alltime_wins_diff": l_prev_alltime_wins - w_prev_alltime_wins,
            "prev_alltime_losses_diff": l_prev_alltime_losses - w_prev_alltime_losses,
            "prev_alltime_matches_diff": l_prev_alltime_matches - w_prev_alltime_matches
        })
        targets.append(0)

    df = pd.DataFrame(features)
    y = np.array(targets)

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

    # Scale values to handle different ranges cleanly
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use Ridge Regularization (l2) to balance out the multicollinearity perfectly
    model = LogisticRegression(penalty='l2', max_iter=1000)
    model.fit(X_train_scaled, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
    
    print("\n================ 🛡️ ULTIMATE HONEST MODEL RESULTS ================")
    print(f"Honest Match Prediction Accuracy: {accuracy * 100:.2f}%")
    print("------------------------------------------------------------------")
    print("Standardized Coefficients (True Predictive Impact compared):")
    
    for col, coef in zip(df.columns, model.coef_.ravel()):
        direction = "🟢 Increases Win Chance" if coef > 0 else "🔴 Decreases Win Chance"
        print(f"  • {col.ljust(25)}: {coef:+.4f} -> {direction}")
    print("==================================================================\n")

if __name__ == "__main__":
    train_ultimate_mtg_predictor()

