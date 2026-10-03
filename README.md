# Netflix Movies Data Science — 2000–2025

Professional but focused 5th-semester Data Science project.

## What it demonstrates
- Data loading and validation
- Data cleaning and type conversion
- Missing-value inspection
- Filtering and feature extraction
- Exploratory Data Analysis (EDA)
- Pandas / NumPy
- Interactive Plotly visualization
- Movie-name search
- Genre, country, rating, duration and year analysis
- CSV export
- Content-based movie recommendation using TF-IDF + cosine similarity
- Top 5–10 similar movie recommendations

## Dataset
The project uses the public `netflix_movies_detailed_up_to_2025.csv` dataset referenced by the CMU 2025 project page. That source describes a 16,000-row movie dataset with fields such as title, release_year, rating, genres, country, popularity, vote_count, budget and revenue.

Important: the dataset is public/TMDb-derived and is not Netflix's private internal database.

## Run
```powershell
py -m pip install -r requirements.txt
py download_data.py
py -m streamlit run app.py
```

Internet is needed once to download the public CSV. After download, the dashboard can run from the local CSV.

The recommendation module uses the selected movie's available metadata and works with the 2000–2025 filtered movie dataset.
