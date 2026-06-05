Predictive model attempt for AGL

* data.csv contains the raw match results for all time
* import.py takes that data and pushes it into a local mongo db
* aggregate_data.mongo combines all the rows into matches (originally each match is two entries - one for winner one for loser - i found it easier to work with them merged)
* generate_alltime_stats.py adds extra fields to the rows to assist with modeling
then there are lots of predictive models. i think v6 was when it started making sense. after that it still kinda works but i was trying to make the leaderboard work which so far has been a failure.
* player_time_patterns.py makes the pretty heat map
* plot_player.py makes a heat map for a specific player

next steps would be to confirm that the model is legit and that i have fixed all the issues with correlated inputs and such. 
after that, figure out how to actually get the leaderboard working. despite all my attempts, it consistently predicts top players as having absurdly high winrates.
