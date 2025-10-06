import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# --- 1. CONFIGURATION AND DATA LOADING ---
st.set_page_config(layout="wide")
st.title("🎬 Netflix Data Analysis Dashboard")
st.markdown("A deep dive into Netflix content distribution, ratings, and popular categories.")
st.markdown("---")

@st.cache_data
def load_data(file_path):
    """Loads, cleans, and transforms the Netflix data."""
    # Load the raw data
    netflix_raw = pd.read_csv(file_path)

    # Copy and clean (dropping NaNs for analyses that require complete records)
    netflix_df = netflix_raw.copy().dropna()

    # Date transformations (as done in the notebook)
    netflix_df["date_added"] = pd.to_datetime(netflix_df['date_added'], format='mixed')
    netflix_df['day_added'] = netflix_df['date_added'].dt.day.astype(int)
    netflix_df['year_added'] = netflix_df['date_added'].dt.year.astype(int)
    netflix_df['month_added'] = netflix_df['date_added'].dt.month

    return netflix_raw, netflix_df

# Load the data - ensure 'netflix_titles.csv' is in the same directory as this script
netflix_raw, netflix_df = load_data("netflix_titles.csv")


# --- 2. DATA SNAPSHOT ---
st.header("1. Data Overview")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Raw Data Snapshot (First 5 Rows)")
    st.dataframe(netflix_raw.head(), use_container_width=True)

with col2:
    st.subheader("Cleaned Data Info")
    st.write(f"**Raw Data Shape:** {netflix_raw.shape[0]} rows, {netflix_raw.shape[1]} columns")
    st.write(f"**Cleaned Data Shape (NaNs Dropped):** {netflix_df.shape[0]} rows, {netflix_df.shape[1]} columns")
    st.markdown("The cleaned dataframe is used for detailed analysis (e.g., Word Clouds).")
st.markdown("---")


# --- 3. TYPE DISTRIBUTION (Movie vs. TV Show) ---
st.header("2. Content Type Distribution")

def plot_type_distribution(df):
    """Generates side-by-side bar and pie plots for content type."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Bar Plot (from code cell 15)
    sns.countplot(x=df['type'], ax=axes[0])
    axes[0].set_title('Count of Content Type', fontsize=16)
    axes[0].set_xlabel('Content Type')
    axes[0].set_ylabel('Count')

    # Pie Chart (from code cell 18)
    labels = ['Movie', 'TV Show']
    size = df['type'].value_counts()
    colors = plt.cm.Wistia(np.linspace(0, 1, 2))
    explode = [0, 0.05]
    axes[1].pie(size, labels=labels, colors=colors, explode=explode, shadow=True, startangle=90, autopct='%1.1f%%')
    axes[1].set_title('Percentage Distribution of Content Type', fontsize=16)
    axes[1].set_ylabel('') # Remove default 'y' label for pie chart

    plt.tight_layout()
    return fig

st.pyplot(plot_type_distribution(netflix_raw))
st.markdown("---")


# --- 4. RATING ANALYSIS ---
st.header("3. Rating Analysis")

def plot_rating_distribution(df):
    """Generates side-by-side bar and pie plots for ratings."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Bar Plot (from code cell 16)
    rating_order = df['rating'].value_counts().index
    sns.countplot(x=df['rating'], order=rating_order, ax=axes[0])
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=70, ha="right", fontsize=12)
    axes[0].set_title('Count of Content by Rating', fontsize=16)

    # Pie Chart (from code cell 19)
    df['rating'].value_counts().plot.pie(autopct='%1.1f%%', shadow=True, ax=axes[1], textprops={'fontsize': 10})
    axes[1].set_ylabel('')
    axes[1].set_title('Percentage Distribution of Content by Rating', fontsize=16)

    plt.tight_layout()
    return fig

def plot_type_vs_rating(df):
    """Generates a bar plot showing the relation between content type and rating."""
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.countplot(x='rating', hue='type', data=df, ax=ax, order=df['rating'].value_counts().index)
    ax.set_title('Relation between Type (Movie/TV Show) and Rating', fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=70, ha='right', fontsize=12)
    plt.tight_layout()
    return fig

st.subheader("3.1. Overall Rating Distribution")
st.pyplot(plot_rating_distribution(netflix_raw))

st.subheader("3.2. Type vs. Rating")
st.pyplot(plot_type_vs_rating(netflix_raw))
st.markdown("---")


# --- 5. WORD CLOUDS ---
st.header("4. Word Cloud Analysis")
st.write("Generating Word Clouds from the **Cleaned Data** to visualize the most frequent entries in key categorical columns.")

@st.cache_data
def generate_wordcloud(df, column_name, title):
    """Generates a WordCloud for a given column."""
    # Using the cleaned dataframe (netflix_df) as per notebook implementation
    # Replacing commas to treat multi-category/multi-name entries as separate words
    text = " ".join(df[column_name].astype(str).str.replace(',', ''))
    wordcloud = WordCloud(
        background_color='white',
        width=1920,
        height=1080
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=20)
    ax.axis('off')
    plt.tight_layout()
    return fig

# Word Cloud selection
wordcloud_option = st.selectbox(
    "Select a column for Word Cloud visualization:",
    ('listed_in (Category)', 'country', 'cast', 'director')
)

# Map selection to function call
wordcloud_map = {
    'listed_in (Category)': ('listed_in', 'Most Popular Categories on Netflix'),
    'country': ('country', 'Most Contributing Countries on Netflix'),
    'cast': ('cast', 'Most Featured Cast Members on Netflix'),
    'director': ('director', 'Most Frequent Directors on Netflix')
}

column, title = wordcloud_map[wordcloud_option]

st.pyplot(generate_wordcloud(netflix_df, column, title))
st.markdown("---")

st.markdown("### End of Analysis")