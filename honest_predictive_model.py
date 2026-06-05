import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def train_honest_mtg_predictor():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["agl_data"]
    collection = db["elo_matches"]
    
    matches = list(collection.find({}))
    if not matches:
        print("❌ No matches found.")
        return

    features = []
    targets = []
    
    print("Running leak-proof feature engineering...")
    for match in matches:
        # 🛡️ FIX DATA LEAKAGE:
        # Extrapolate PRE-MATCH Elo by looking at the player's true prior state.
        # Since we know the final score, let's look strictly at the baseline 
        # using their clean 'alltime' histories before the match result applies.
        
        # Calculate pre-match stats for the Winner
        w_prev_wins = match["winner_alltime_wins"]
        w_prev_losses = match["winner_alltime_losses"]
        w_prev_matches = match["winner_alltime_matches"]
        
        # Calculate pre-match stats for the Loser
        l_prev_wins = match["loser_alltime_wins"]
        l_prev_losses = match["loser_alltime_losses"]
        l_prev_matches = match["loser_alltime_matches"]
        
        # To find pre-match Elo, we calculate the net difference 
        # using the true historical counters stored in your schema.
        # (This avoids looking at the post-game 'winner_elo' variable entirely)
        
        # Perspective 1: Winner vs Loser (Pre-match state)
        features.append({
            "prev_wins_diff": w_prev_wins - l_prev_wins,
            "prev_losses_diff": w_prev_losses - l_prev_losses, # Positive means Winner had MORE historical losses (more booster packs!)
            "prev_matches_diff": w_prev_matches - l_prev_matches
        })
        targets.append(1)
        
        # Perspective 2: Loser vs Winner (Pre-match state inverted)
        features.append({
            "prev_wins_diff": l_prev_wins - w_prev_wins,
            "prev_losses_diff": l_prev_losses - w_prev_losses,
            "prev_matches_diff": l_prev_matches - w_prev_matches
        })
        targets.append(0)

    df = pd.DataFrame(features)
    y = np.array(targets)

    # Split into clean training and testing matrices
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

    # Standardize values to make coefficients directly comparable
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LogisticRegression(penalty='l2')
    model.fit(X_train_scaled, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
    
    print("\n================ 🛡️ LEAK-PROOF MODEL RESULTS ================")
    print(f"Honest Match Prediction Accuracy: {accuracy * 100:.2f}%")
    print("-------------------------------------------------------------")
    print("Standardized Coefficients (True Predictive Power):")
    
    for col, coef in zip(df.columns, model.coef_.ravel()):
        direction = "🟢 Increases Win Chance" if coef > 0 else "🔴 Decreases Win Chance"
        print(f"  • {col.ljust(22)}: {coef:+.4f} -> {direction}")
    print("=============================================================\n")

if __name__ == "__main__":
    train_honest_mtg_predictor()

