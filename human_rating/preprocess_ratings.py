import pandas as pd
# Load the Excel file
file_path = "/Users/valerieluthi/PycharmProjects/rag_carasavapa/human_rating/RatingCVs_combined.xlsx"
df_ratings = pd.read_excel(file_path)
# drop empty columns
df_ratings.drop(columns=['Rating Valerie', 'Comment Valerie'], inplace=True)
# convert binary rating to "Yes" = 1 and "No" = 0
df_ratings["Ratings combined"] = df_ratings["Ratings combined"].map({"Yes":1, "No":0})
# check conversion
print(df_ratings[["Ratings combined"]].head())
# Example: % of "Yes" (hires) overall
percent_yes = df_ratings["Ratings combined"].mean() * 100
print(f"Percentage of 'Yes' ratings: {percent_yes:.1f}%")
# Example: average per role
avg_by_level = df_ratings.groupby("JD Level")["Ratings combined"].mean()
print(avg_by_level)
# convert to float
df_ratings["Ratings combined"] = pd.to_numeric(df_ratings["Ratings combined"], errors="coerce")
df_ratings["Ratings combined"] = df_ratings["Ratings combined"].fillna(0).astype(int)
print(df_ratings.info())
# save
df_ratings.to_csv("../human_rating/ratings_combined_clean.csv", index=False)