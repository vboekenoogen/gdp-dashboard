import streamlit as st
import pandas as pd
import numpy as np
import io
import re
from collections import Counter
import chardet
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER analyzer
vader_analyzer = SentimentIntensityAnalyzer()

def detect_encoding(file_bytes):
    """Detect file encoding automatically"""
    result = chardet.detect(file_bytes)
    return result['encoding']

def read_csv_with_encoding(uploaded_file):
    """Read CSV with automatic encoding detection"""
    try:
        # Read file bytes
        file_bytes = uploaded_file.read()
        
        # Detect encoding
        encoding = detect_encoding(file_bytes)
        st.info(f"Detected encoding: {encoding}")
        
        # Reset file pointer
        uploaded_file.seek(0)
        
        # Try detected encoding first
        try:
            df = pd.read_csv(uploaded_file, encoding=encoding)
            return df
        except:
            # Fallback encodings
            encodings_to_try = ['utf-8', 'utf-16', 'iso-8859-1', 'cp1252', 'latin1']
            
            for enc in encodings_to_try:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=enc)
                    st.info(f"Successfully read with encoding: {enc}")
                    return df
                except:
                    continue
            
            raise Exception("Could not read file with any supported encoding")
            
    except Exception as e:
        st.error(f"Error reading CSV: {str(e)}")
        return None

def count_words(text):
    """Count words in text"""
    if pd.isna(text):
        return 0
    return len(str(text).split())

def get_classifier_columns(df):
    """Automatically detect classifier columns with 'has_' prefix"""
    return [col for col in df.columns if col.startswith('has_')]

def analyze_sentiment_textblob(text):
    """Analyze sentiment using TextBlob"""
    if pd.isna(text):
        return 'Neutral', 0, 0
    
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    
    # Classify sentiment
    if polarity > 0.1:
        label = 'Positive'
    elif polarity < -0.1:
        label = 'Negative'
    else:
        label = 'Neutral'
    
    # Count positive and negative words (simplified approach)
    words = str(text).lower().split()
    
    # Basic positive/negative word lists
    positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
                     'perfect', 'love', 'best', 'awesome', 'brilliant', 'outstanding',
                     'superb', 'terrific', 'marvelous', 'exceptional', 'incredible'}
    
    negative_words = {'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'disgusting',
                     'disappointing', 'poor', 'useless', 'pathetic', 'dreadful', 'appalling',
                     'atrocious', 'abysmal', 'deplorable', 'wretched'}
    
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    total_words = len(words)
    
    pos_pct = (positive_count / total_words * 100) if total_words > 0 else 0
    neg_pct = (negative_count / total_words * 100) if total_words > 0 else 0
    
    return label, pos_pct, neg_pct

def analyze_sentiment_vader(text):
    """Analyze sentiment using VADER"""
    if pd.isna(text):
        return 'Neutral', 0, 0
    
    scores = vader_analyzer.polarity_scores(str(text))
    compound = scores['compound']
    
    # Classify sentiment based on compound score
    if compound >= 0.05:
        label = 'Positive'
    elif compound <= -0.05:
        label = 'Negative'
    else:
        label = 'Neutral'
    
    # Use VADER's positive and negative scores as percentages
    pos_pct = scores['pos'] * 100
    neg_pct = scores['neg'] * 100
    
    return label, pos_pct, neg_pct

def calculate_classifier_percentage(row, classifier_cols, total_words):
    """Calculate percentage of classifier-flagged words"""
    if total_words == 0:
        return 0
    
    # Count how many classifier flags are set to 1
    flagged_classifiers = sum(row[col] for col in classifier_cols if pd.notna(row[col]) and row[col] == 1)
    
    # This is a simplified calculation - in reality, you might want to 
    # count actual words that triggered the classifiers
    # For now, we'll assume each flagged classifier represents some proportion of words
    classifier_word_ratio = flagged_classifiers / len(classifier_cols) if classifier_cols else 0
    
    return classifier_word_ratio * 100

def process_statement_level_metrics(df, classifier_cols, sentiment_method):
    """Process statement-level metrics"""
    results = []
    
    for idx, row in df.iterrows():
        statement = row['Statement']
        total_words = count_words(statement)
        
        # Calculate classifier word percentage
        classifier_pct = calculate_classifier_percentage(row, classifier_cols, total_words)
        
        # Analyze sentiment
        if sentiment_method == 'TextBlob':
            sentiment_label, pos_pct, neg_pct = analyze_sentiment_textblob(statement)
        else:  # VADER
            sentiment_label, pos_pct, neg_pct = analyze_sentiment_vader(statement)
        
        results.append({
            'ID': row['ID'],
            'Statement': statement,
            'total_words': total_words,
            '%_classifier_words': round(classifier_pct, 2),
            'sentiment_label': sentiment_label,
            '%_positive_words': round(pos_pct, 2),
            '%_negative_words': round(neg_pct, 2)
        })
    
    return pd.DataFrame(results)

def aggregate_by_id(statement_df):
    """Aggregate metrics by ID"""
    agg_results = []
    
    for id_val in statement_df['ID'].unique():
        id_data = statement_df[statement_df['ID'] == id_val]
        
        agg_results.append({
            'ID': id_val,
            'total_statements': len(id_data),
            'total_words': id_data['total_words'].sum(),
            'avg_words_per_statement': round(id_data['total_words'].mean(), 2),
            'avg_%_classifier_words': round(id_data['%_classifier_words'].mean(), 2),
            'avg_%_positive_words': round(id_data['%_positive_words'].mean(), 2),
            'avg_%_negative_words': round(id_data['%_negative_words'].mean(), 2)
        })
    
    return pd.DataFrame(agg_results)

def create_download_link(df, filename, link_text):
    """Create a download link for DataFrame"""
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    
    return st.download_button(
        label=link_text,
        data=csv_data,
        file_name=filename,
        mime='text/csv',
        key=filename
    )

def main():
    st.set_page_config(
        page_title="Social Media Statement Analyzer",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Social Media Statement Analyzer")
    st.markdown("Upload a CSV file with social media statements to analyze sentiment and classifier metrics.")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    sentiment_method = st.sidebar.selectbox(
        "Sentiment Analysis Method",
        ["VADER", "TextBlob"],
        help="Choose the sentiment analysis library to use"
    )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=['csv'],
        help="CSV must contain columns: ID, Statement, and classifier columns prefixed with 'has_'"
    )
    
    if uploaded_file is not None:
        # Read CSV with encoding detection
        df = read_csv_with_encoding(uploaded_file)
        
        if df is not None:
            st.success("✅ File uploaded successfully!")
            
            # Validate required columns
            required_cols = ['ID', 'Statement']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
                return
            
            # Display basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                st.metric("Unique IDs", df['ID'].nunique())
            with col3:
                classifier_cols = get_classifier_columns(df)
                st.metric("Classifier Columns", len(classifier_cols))
            
            # Show classifier columns found
            if classifier_cols:
                st.info(f"**Detected classifier columns:** {', '.join(classifier_cols)}")
            else:
                st.warning("⚠️ No classifier columns found (columns starting with 'has_')")
            
            # Preview data
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10))
            
            # Process button
            if st.button("🚀 Analyze Data", type="primary"):
                with st.spinner("Processing data..."):
                    # Process statement-level metrics
                    statement_metrics = process_statement_level_metrics(
                        df, classifier_cols, sentiment_method
                    )
                    
                    # Aggregate by ID
                    id_metrics = aggregate_by_id(statement_metrics)
                    
                    # Display results
                    st.subheader("📊 Results")
                    
                    tab1, tab2 = st.tabs(["Statement Level Metrics", "ID Level Aggregated Metrics"])
                    
                    with tab1:
                        st.markdown("**Statement Level Word Metrics**")
                        st.dataframe(statement_metrics)
                        
                        # Download button
                        create_download_link(
                            statement_metrics,
                            "statement_level_word_metrics.csv",
                            "📥 Download Statement Level Metrics"
                        )
                    
                    with tab2:
                        st.markdown("**ID Level Aggregated Metrics**")
                        st.dataframe(id_metrics)
                        
                        # Download button
                        create_download_link(
                            id_metrics,
                            "id_level_aggregated_metrics.csv",
                            "📥 Download ID Level Metrics"
                        )
                    
                    st.success("✅ Analysis complete! Download your results using the buttons above.")
    
    # Instructions
    st.sidebar.markdown("---")
    st.sidebar.header("📝 Instructions")
    st.sidebar.markdown("""
    **Required CSV Format:**
    - `ID`: Unique post identifier
    - `Statement`: Text to analyze
    - `has_*`: Binary classifier columns (0/1)
    
    **Example:**
    ```csv
    ID,Statement,has_inquiring
    123,"How can I help?",1
    123,"That's great!",0
    ```
    """)

if __name__ == "__main__":
    main()
