import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def calculate_smoothed_win_rate(wins, matches, prior_weight=2):
    """
    Calculates a regularized win rate using Laplace smoothing.
    Prevents 1-0 players from having a 100% win rate, and handles 0 matches safely.
    Assumes an average baseline win rate of 50%.
    """
    if matches == 0:
        return 0.50 # A fresh player defaults to a 50% expected win rate
    
    # Add 'prior_weight' wins and losses to anchor the percentage
    smoothed_wins = wins + prior_weight
    smoothed_matches = matches + (prior_weight * 2)
    return smoothed_wins / smoothed_matches

def train_rate_based_predictor():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["agl_data"]
    collection = db["elo_matches"]
    
    matches = list(collection.find({}))
    if not matches:
        print("❌ No matches found in 'elo_matches'.")
        return

    features = []
    targets = []
    
    print("Calculating leak-proof smoothed win rates for current league and career...")
    for match in matches:
        # --- 1. Fetch Raw Pre-Match Career Counters ---
        w_alltime_w = match["winner_alltime_wins"]
        w_alltime_m = match["winner_alltime_matches"]
        l_alltime_w = match["loser_alltime_wins"]
        l_alltime_m = match["loser_alltime_matches"]
        
        # --- 2. Fetch Raw Pre-Match League Counters ---
        w_league_w = match.get("winner_n_wins", 0) - 1
        w_league_m = match.get("winner_n_matches", 0) - 1
        l_league_w = match.get("loser_n_wins", 0)
        l_league_m = match.get("loser_n_matches", 0) - 1

        # --- 3. Compute Regularized Percentages ---
        w_alltime_rate = calculate_smoothed_win_rate(w_alltime_w, w_alltime_m)
        l_alltime_rate = calculate_smoothed_win_rate(l_alltime_w, l_alltime_m)
        
        w_league_rate = calculate_smoothed_win_rate(w_league_w, w_league_m)
        l_league_rate = calculate_smoothed_win_rate(l_league_w, l_league_m)
        
        # --- 4. Build Perspectives Matrix ---
        # Perspective 1: Winner vs Loser
        features.append({
            "alltime_winrate_diff": w_alltime_rate - l_alltime_rate,
            "league_winrate_diff": w_league_rate - l_league_rate,
            # We preserve total match experience difference to let the model capture 
            # if pure volume/practice grants any subtle advantage separate from win rate.
            "alltime_experience_diff": w_alltime_m - l_alltime_m,
            "league_experience_diff": w_league_m - l_league_m
        })
        targets.append(1)
        
        # Perspective 2: Loser vs Winner (Inverted)
        features.append({
            "alltime_winrate_diff": l_alltime_rate - w_alltime_rate,
            "league_winrate_diff": l_league_rate - w_league_rate,
            "alltime_experience_diff": l_alltime_m - w_alltime_m,
            "league_experience_diff": l_league_m - w_league_m
        })
        targets.append(0)

    df = pd.DataFrame(features)
    y = np.array(targets)

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

    # Scale to normalize raw match volume vs percentage metrics
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a clean regularized model
    model = LogisticRegression(penalty='l2')
    model.fit(X_train_scaled, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
    
    print("\n================ 🛡️ EFFICIENCY RATE MODEL RESULTS ================")
    print(f"Honest Match Prediction Accuracy: {accuracy * 100:.2f}%")
    print("------------------------------------------------------------------")
    print("Standardized Coefficients (True Efficiency Power compared):")
    
    for col, coef in zip(df.columns, model.coef_.ravel()):
        direction = "🟢 Increases Win Chance" if coef > 0 else "🔴 Decreases Win Chance"
        print(f"  • {col.ljust(25)}: {coef:+.4f} -> {direction}")
    print("==================================================================\n")

if __name__ == "__main__":
    train_rate_based_predictor()

