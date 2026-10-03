
import re
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Netflix Movies | Data Science",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {background:#0b0d10;}
[data-testid="stSidebar"] {background:#11151a;}
.block-container {max-width:1450px; padding-top:1.5rem;}
h1,h2,h3 {letter-spacing:-.4px;}
[data-testid="stMetric"] {background:#14191f;border:1px solid #252c34;border-radius:14px;padding:12px;}
.stButton>button {border-radius:9px;}
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "netflix_movies_detailed_up_to_2025.csv"
URL = "https://raw.githubusercontent.com/Raffaele2004/netflix-data-analysis/main/netflix_movies_detailed_up_to_2025.csv"

@st.cache_data(show_spinner=False)
def load_data():
    if DATA.exists():
        return pd.read_csv(DATA, sep=";", encoding="utf-8-sig")
    r = requests.get(URL, timeout=60)
    r.raise_for_status()
    DATA.parent.mkdir(exist_ok=True)
    DATA.write_bytes(r.content)
    return pd.read_csv(DATA, sep=";", encoding="utf-8-sig")

try:
    raw = load_data()
except Exception as e:
    st.error("Dataset could not be loaded.")
    st.code(str(e))
    st.info("Run:  py download_data.py   and then restart the dashboard.")
    st.stop()

df = raw.copy()
df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

# Standardize likely source column names.
rename = {}
for a,b in [
    ("name","title"),("movie_title","title"),("releaseyear","release_year"),
    ("voteaverage","vote_average"),("votecount","vote_count"),
    ("dateadded","date_added"),("listedin","genres")
]:
    if a in df.columns and b not in df.columns: rename[a]=b
df = df.rename(columns=rename)

# Data cleaning / type conversion
for c in ["release_year","rating","popularity","vote_count","vote_average","budget","revenue"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

if "release_year" not in df.columns:
    st.error("The dataset does not contain a release_year column.")
    st.stop()

df = df[df["release_year"].between(2000, 2025)].copy()

# Keep movie records if a type column exists.
if "type" in df.columns:
    movie_mask = df["type"].astype(str).str.lower().eq("movie")
    if movie_mask.any():
        df = df[movie_mask].copy()

# Clean strings without destroying missingness.
for c in ["title","director","cast","country","genres","language","description"]:
    if c in df.columns:
        df[c] = df[c].fillna("Unknown").astype(str).str.strip()

if "title" not in df.columns:
    st.error("Movie title column is missing from the dataset.")
    st.stop()

# Duration parser
if "duration" in df.columns:
    df["duration_min"] = pd.to_numeric(
        df["duration"].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0],
        errors="coerce"
    )


@st.cache_data(show_spinner=False)
def build_recommender(data):
    """Build a lightweight content-based recommender from available movie metadata."""
    d = data.copy().reset_index(drop=True)
    fields = [c for c in [
        "genres", "director", "cast", "country", "language", "description",
        "rating"
    ] if c in d.columns]
    if not fields or "title" not in d.columns or len(d) < 2:
        return None, None, None

    for c in fields:
        d[c] = d[c].fillna("").astype(str)

    # Repeat genre/rating lightly so the recommender pays attention to these signals.
    parts = []
    for _, row in d.iterrows():
        text = " ".join(row[c] for c in fields)
        if "genres" in d.columns:
            text += " " + row["genres"] + " " + row["genres"]
        if "rating" in d.columns:
            text += " " + row["rating"]
        parts.append(text.lower())

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=12000
    )
    matrix = vectorizer.fit_transform(parts)
    return d, matrix, vectorizer

def recommend_movies(data, selected_title, n=10):
    built = build_recommender(data)
    if built[0] is None:
        return pd.DataFrame()
    d, matrix, _ = built
    matches = d.index[d["title"].str.casefold() == selected_title.casefold()].tolist()
    if not matches:
        matches = d.index[d["title"].str.contains(
            re.escape(selected_title), case=False, na=False
        )].tolist()
    if not matches:
        return pd.DataFrame()

    idx = matches[0]
    scores = cosine_similarity(matrix[idx], matrix).ravel()
    order = np.argsort(scores)[::-1]
    order = [i for i in order if i != idx][:n]

    cols = [c for c in [
        "title", "release_year", "rating", "genres", "country",
        "director", "duration", "vote_average", "vote_count", "description"
    ] if c in d.columns]
    out = d.loc[order, cols].copy()
    out.insert(0, "similarity_score", scores[order])
    out["similarity_score"] = (out["similarity_score"] * 100).round(1)
    return out

st.sidebar.title("🎬 Netflix Analytics")
st.sidebar.caption("Data Science • EDA • 2000–2025")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "EDA & Trends", "Genre & Country", "Movie Search", "🤖 Recommendations", "Data Quality"]
)

st.sidebar.divider()
st.sidebar.subheader("Filters")

ymin = int(df["release_year"].min()) if len(df) else 2000
ymax = int(df["release_year"].max()) if len(df) else 2025
yr = st.sidebar.slider("Release year", 2000, 2025, (max(2000,ymin), min(2025,ymax)))
f = df[df["release_year"].between(yr[0], yr[1])].copy()

if "genres" in f.columns:
    genre_series = f["genres"].str.split(r",\s*").explode().str.strip()
    genre_options = sorted([x for x in genre_series.dropna().unique() if x and x != "Unknown"])
    chosen_genres = st.sidebar.multiselect("Genre", genre_options)
    if chosen_genres:
        f = f[f["genres"].apply(lambda x: any(g.lower() in x.lower() for g in chosen_genres))]

if "rating" in f.columns:
    rating_options = sorted(f["rating"].dropna().astype(str).unique())
    chosen_ratings = st.sidebar.multiselect("Rating", rating_options)
    if chosen_ratings:
        f = f[f["rating"].astype(str).isin(chosen_ratings)]

st.title("🎬 Netflix Movies Data Science Dashboard")
st.caption("Clean → Explore → Visualize → Interpret | Movie release years: 2000–2025")

# KPI row
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Movies", f"{len(f):,}")
k2.metric("Years", f"{f['release_year'].nunique():,}")
k3.metric("Countries", f"{f['country'].nunique():,}" if "country" in f else "—")
k4.metric("Genres", f"{f['genres'].nunique():,}" if "genres" in f else "—")
if "duration_min" in f:
    k5.metric("Avg. Duration", f"{f['duration_min'].mean():.0f} min" if f["duration_min"].notna().any() else "—")
else:
    k5.metric("Columns", f"{f.shape[1]:,}")

st.divider()

if page == "Overview":
    st.subheader("📊 Dataset Overview")
    a,b = st.columns(2)
    with a:
        y = f.groupby("release_year").size().reset_index(name="movies")
        fig = px.area(y, x="release_year", y="movies", markers=True,
                      title="Movies by Release Year")
        fig.update_layout(margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with b:
        if "rating" in f:
            r = f["rating"].value_counts().head(12).reset_index()
            r.columns=["rating","movies"]
            fig = px.bar(r, x="rating", y="movies", title="Rating Distribution",
                         text_auto=True)
            fig.update_layout(margin=dict(l=10,r=10,t=50,b=10))
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("📌 Quick Data Insights")
    insights=[]
    if len(f):
        peak_year = int(f["release_year"].value_counts().idxmax())
        insights.append(f"Highest movie count in the selected data: **{peak_year}**.")
    if "duration_min" in f and f["duration_min"].notna().any():
        insights.append(f"Average movie duration in the selection: **{f['duration_min'].mean():.1f} minutes**.")
    if "genres" in f:
        topg = f["genres"].str.split(r",\s*").explode().value_counts()
        if len(topg): insights.append(f"Most represented genre tag: **{topg.index[0]}**.")
    for x in insights:
        st.info(x)

elif page == "EDA & Trends":
    st.subheader("📈 Exploratory Data Analysis")
    a,b=st.columns(2)
    with a:
        y=f.groupby("release_year").size().reset_index(name="movies")
        fig=px.line(y,x="release_year",y="movies",markers=True,title="Year-wise Trend")
        st.plotly_chart(fig,use_container_width=True)
    with b:
        if "duration_min" in f:
            d=f["duration_min"].dropna()
            fig=px.histogram(d,x="duration_min",nbins=35,title="Movie Duration Distribution")
            st.plotly_chart(fig,use_container_width=True)
    if "vote_average" in f and "vote_count" in f:
        s=f[["vote_average","vote_count"]].dropna()
        if len(s):
            fig=px.scatter(s,x="vote_count",y="vote_average",opacity=.55,
                           log_x=True,title="Votes vs Average Rating",
                           labels={"vote_count":"Vote Count","vote_average":"Average Rating"})
            st.plotly_chart(fig,use_container_width=True)
    if "budget" in f and "revenue" in f:
        fin=f[["budget","revenue"]].replace(0,np.nan).dropna()
        if len(fin):
            fin=fin.sample(min(len(fin),4000),random_state=42)
            fig=px.scatter(fin,x="budget",y="revenue",log_x=True,log_y=True,
                           title="Budget vs Revenue")
            st.plotly_chart(fig,use_container_width=True)

elif page == "Genre & Country":
    st.subheader("🌍 Genre & Country Analysis")
    a,b=st.columns(2)
    with a:
        if "genres" in f:
            g=f["genres"].str.split(r",\s*").explode().str.strip()
            g=g[g!="Unknown"].value_counts().head(15).sort_values()
            fig=px.bar(g,x=g.values,y=g.index,orientation="h",title="Top Genres")
            st.plotly_chart(fig,use_container_width=True)
    with b:
        if "country" in f:
            c=f["country"].str.split(r",\s*").explode().str.strip()
            c=c[c!="Unknown"].value_counts().head(15).sort_values()
            fig=px.bar(c,x=c.values,y=c.index,orientation="h",title="Top Countries")
            st.plotly_chart(fig,use_container_width=True)

    if "genres" in f and "rating" in f:
        gdf=f[["genres","rating"]].dropna().copy()
        gdf["genre"]=gdf["genres"].str.split(r",\s*")
        gdf=gdf.explode("genre")
        top=gdf["genre"].value_counts().head(10).index
        gdf=gdf[gdf["genre"].isin(top)]
        if len(gdf):
            fig=px.box(gdf,x="genre",y="rating",title="Rating Pattern Across Top Genres")
            st.plotly_chart(fig,use_container_width=True)

elif page == "Movie Search":
    st.subheader("🔎 Movie Search")
    q=st.text_input("Search by movie name", placeholder="Type a movie name, e.g. Inception...")
    result=f.copy()
    if q.strip():
        result=result[result["title"].str.contains(q.strip(),case=False,na=False)]
    st.write(f"**{len(result):,} matching movie(s)**")
    if len(result):
        cols=[c for c in ["title","release_year","rating","genres","country","director","duration","vote_average","vote_count"] if c in result.columns]
        st.dataframe(result[cols].sort_values("title"),use_container_width=True,hide_index=True)
        st.download_button("⬇️ Download search results",
                           result.to_csv(index=False).encode("utf-8"),
                           "movie_search_results.csv","text/csv")
    else:
        st.warning("No movie found. Try a shorter title or remove some filters.")

elif page == "🤖 Recommendations":
    st.subheader("🤖 Netflix Movie Recommendation System")
    st.caption("Content-based recommendation using TF-IDF and cosine similarity.")

    rec_source = f.copy()
    rec_source = rec_source.drop_duplicates(subset=["title"], keep="first")
    titles = sorted(rec_source["title"].dropna().astype(str).unique().tolist())

    if len(titles) < 2:
        st.warning("Not enough movie records are available for recommendations.")
    else:
        left, right = st.columns([2, 1])
        with left:
            selected = st.selectbox(
                "🎬 Select a movie",
                titles,
                index=0,
                help="Choose a movie to find similar movies from the current 2000–2025 filtered dataset."
            )
        with right:
            top_n = st.slider("Number of recommendations", 5, 10, 5)

        selected_row = rec_source[
            rec_source["title"].astype(str).str.casefold() == selected.casefold()
        ].head(1)

        if len(selected_row):
            show_cols = [c for c in [
                "title", "release_year", "rating", "genres", "country",
                "director", "duration", "vote_average"
            ] if c in selected_row.columns]
            st.dataframe(
                selected_row[show_cols],
                use_container_width=True,
                hide_index=True
            )

        with st.spinner("Finding similar movies..."):
            recommendations = recommend_movies(rec_source, selected, top_n)

        if len(recommendations):
            st.markdown("### ✨ Recommended Movies")
            display = recommendations.copy()
            display["similarity_score"] = display["similarity_score"].map(lambda x: f"{x:.1f}%")
            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "similarity_score": st.column_config.TextColumn(
                        "Similarity", help="Metadata similarity score"
                    ),
                    "description": st.column_config.TextColumn("Description", width="large")
                }
            )

            chart = recommendations[["title", "similarity_score"]].sort_values(
                "similarity_score"
            )
            fig = px.bar(
                chart, x="similarity_score", y="title", orientation="h",
                title="Recommendation Similarity"
            )
            fig.update_layout(
                xaxis_title="Similarity (%)",
                yaxis_title="Movie",
                margin=dict(l=10, r=10, t=50, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

            st.info(
                "How it works: movie metadata such as genre, director, cast, "
                "country, language, rating and description is converted into "
                "TF-IDF features. Cosine similarity is then used to identify "
                "movies with similar content."
            )
        else:
            st.warning("Could not generate recommendations for the selected movie.")

elif page == "Data Quality":
    st.subheader("🧹 Data Quality & Preprocessing")
    q1,q2,q3=st.columns(3)
    q1.metric("Rows after filters",f"{len(f):,}")
    q2.metric("Columns",f"{f.shape[1]:,}")
    q3.metric("Duplicate rows",f"{f.duplicated().sum():,}")
    st.write("Missing values before display/analysis:")
    miss=f.isna().sum().sort_values(ascending=False).reset_index()
    miss.columns=["column","missing"]
    st.dataframe(miss,use_container_width=True,hide_index=True)
    st.write("Data types:")
    types=pd.DataFrame({"column":f.columns,"dtype":[str(x) for x in f.dtypes]})
    st.dataframe(types,use_container_width=True,hide_index=True)
    st.download_button("⬇️ Download cleaned filtered data",
                       f.to_csv(index=False).encode("utf-8"),
                       "netflix_movies_cleaned.csv","text/csv")

st.divider()
st.caption("Data Science workflow: data loading • cleaning • preprocessing • EDA • visualization • interpretation")
