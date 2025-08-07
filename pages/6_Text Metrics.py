import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from collections import Counter
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Classifier Word Metrics Tool",
    page_icon="📊",
    layout="wide"
)

st.title("🔍 Classifier Word Metrics Tool")
st.markdown("Convert binary classification to continuous measures with word-level analysis")

# Sidebar for file uploads
st.sidebar.header("📂 Data Upload")

# Input method selection
input_method = st.sidebar.radio(
    "📂 Choose Input Method:",
    ["Ground Truth Data", "IG Posts Data"],
    help="Select your data source type"
)

if input_method == "Ground Truth Data":
    data_file = st.sidebar.file_uploader(
        "Upload Ground Truth CSV", 
        type=['csv'], 
        key="ground_truth",
        help="Upload CSV with columns: ID, Turn, Statement, Mode (binary 0/1)"
    )
    st.sidebar.info("✅ Expected columns: ID, Turn, Statement, Mode")
    
else:  # IG Posts Data
    data_file = st.sidebar.file_uploader(
        "Upload IG Posts CSV", 
        type=['csv'], 
        key="ig_posts", 
        help="Upload Instagram posts data with text content and engagement metrics"
    )
    st.sidebar.info("ℹ️ Will process post content and generate classifications")

# Initialize session state for processed data
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

def safe_read_csv(file, encodings=['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']):
    """
    Safely read CSV file with multiple encoding attempts
    """
    for encoding in encodings:
        try:
            # Reset file pointer
            file.seek(0)
            df = pd.read_csv(file, encoding=encoding)
            st.success(f"✅ File loaded successfully with {encoding} encoding")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            st.error(f"Error with {encoding}: {str(e)}")
            continue
    
    # If all encodings fail
    st.error("❌ Could not read file with any supported encoding. Please check your file format.")
    return None

def detect_ig_posts_structure(df):
    """
    Detect and adapt IG posts data structure
    """
    # Common column name variations for IG posts
    text_columns = ['caption', 'text', 'content', 'post_text', 'description', 'message']
    id_columns = ['id', 'post_id', 'ig_id', 'shortcode', 'post_shortcode']
    
    # Find text column
    text_col = None
    for col in df.columns:
        if col.lower() in text_columns or 'text' in col.lower() or 'caption' in col.lower():
            text_col = col
            break
    
    # Find ID column  
    id_col = None
    for col in df.columns:
        if col.lower() in id_columns or 'id' in col.lower():
            id_col = col
            break
    
    if not text_col:
        st.error("❌ Could not find text/caption column in IG posts data")
        st.write("Available columns:", list(df.columns))
        return None, None
        
    if not id_col:
        st.warning("⚠️ Could not find ID column, will generate sequential IDs")
        
    return text_col, id_col

def convert_ig_posts_to_ground_truth(df, text_col, id_col=None):
    """
    Convert IG posts data to ground truth format
    """
    processed_rows = []
    
    for idx, row in df.iterrows():
        # Get post ID
        post_id = row[id_col] if id_col else f"post_{idx:04d}"
        
        # Get text content
        text_content = row[text_col] if pd.notna(row[text_col]) else ""
        
        # Split text into sentences/statements
        sentences = re.split(r'[.!?]+', str(text_content))
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # If no sentences found, use the whole text as one statement
        if not sentences:
            sentences = [str(text_content)]
        
        # Create statements with auto-classification
        for turn, statement in enumerate(sentences, 1):
            if statement.strip():
                # Auto-classify based on personalization keywords
                keywords = extract_personalized_keywords(statement)
                mode = 1 if keywords else 0
                
                processed_rows.append({
                    'ID': str(post_id),
                    'Turn': turn,
                    'Statement': statement.strip(),
                    'Mode': mode,
                    'auto_classified': True
                })
    
    return pd.DataFrame(processed_rows)
    """Clean and preprocess text"""
    if pd.isna(text) or text == "":
        return ""
    # Remove extra whitespace and convert to lowercase
    text = re.sub(r'\s+', ' ', str(text)).strip().lower()
    return text

def extract_personalized_keywords(text):
    """Extract keywords that indicate personalization"""
    personalized_keywords = [
        'custom', 'personalized', 'tailored', 'bespoke', 'individual', 'unique',
        'customized', 'made for you', 'personal', 'special', 'exclusive',
        'one-of-a-kind', 'handmade', 'curated', 'designed for', 'your style',
        'perfect for you', 'just for you', 'match your', 'fit your'
    ]
    
    if pd.isna(text) or text == "":
        return []
    
    text_lower = str(text).lower()
    found_keywords = []
    
    for keyword in personalized_keywords:
        if keyword in text_lower:
            found_keywords.append(keyword)
    
    return found_keywords

def calculate_personalization_strength(text, binary_classification):
    """Calculate continuous personalization strength"""
    if pd.isna(text) or text == "":
        return 0.0
    
    keywords = extract_personalized_keywords(text)
    keyword_score = len(keywords) * 0.2  # Each keyword adds 0.2
    
    # Base score from binary classification
    base_score = float(binary_classification)
    
    # Combine scores (max 1.0)
    total_score = min(base_score + keyword_score, 1.0)
    
    return total_score

def analyze_sentiment(text):
    """Analyze sentiment of text"""
    if pd.isna(text) or text == "":
        return {'polarity': 0, 'subjectivity': 0, 'sentiment': 'neutral'}
    
    try:
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'sentiment': sentiment
        }
    except:
        return {'polarity': 0, 'subjectivity': 0, 'sentiment': 'neutral'}

def process_data():
    """Main data processing function"""
    if not ground_truth_file:
        st.warning("Please upload the Ground Truth data file to proceed.")
        return None
    
    try:
        # Load data
        with st.spinner("Loading ground truth data..."):
            ground_truth_df = pd.read_csv(ground_truth_file)
            ig_posts_df = pd.read_csv(ig_posts_file) if ig_posts_file else None
        
        # Process ground truth data
        with st.spinner("Processing ground truth data..."):
            ground_truth_df['statement_clean'] = ground_truth_df['Statement'].apply(preprocess_text)
            ground_truth_df['word_count'] = ground_truth_df['statement_clean'].apply(lambda x: len(str(x).split()) if x else 0)
            
            # Calculate personalization strength
            ground_truth_df['personalization_strength'] = ground_truth_df.apply(
                lambda row: calculate_personalization_strength(row['Statement'], row['Mode']), axis=1
            )
            
            # Extract personalized keywords
            ground_truth_df['personalized_keywords'] = ground_truth_df['Statement'].apply(extract_personalized_keywords)
            ground_truth_df['keyword_count'] = ground_truth_df['personalized_keywords'].apply(len)
            
            # Sentiment analysis
            sentiment_data = ground_truth_df['Statement'].apply(analyze_sentiment)
            ground_truth_df['sentiment_polarity'] = sentiment_data.apply(lambda x: x['polarity'])
            ground_truth_df['sentiment_subjectivity'] = sentiment_data.apply(lambda x: x['subjectivity'])
            ground_truth_df['sentiment_category'] = sentiment_data.apply(lambda x: x['sentiment'])
        
        # Create statement-level metrics
        with st.spinner("Creating statement-level metrics..."):
            statement_metrics = ground_truth_df.copy()
            statement_metrics['personalized_word_percentage'] = (
                statement_metrics['keyword_count'] / statement_metrics['word_count'].replace(0, 1) * 100
            )
        
        # Aggregate to ID level
        with st.spinner("Aggregating to ID level..."):
            id_level_metrics = ground_truth_df.groupby('ID').agg({
                'Mode': 'sum',  # Total personalized statements
                'Turn': 'count',  # Total statements
                'word_count': 'sum',  # Total words
                'keyword_count': 'sum',  # Total personalized keywords
                'personalization_strength': 'mean',  # Average strength
                'sentiment_polarity': 'mean',  # Average sentiment
                'sentiment_subjectivity': 'mean',
                'Statement': lambda x: ' '.join(x)  # Concatenate all statements
            }).reset_index()
            
            # Rename columns
            id_level_metrics.columns = [
                'ID', 'personalized_statements_count', 'total_statements', 'total_words',
                'personalized_keywords_count', 'avg_personalization_strength',
                'avg_sentiment_polarity', 'avg_sentiment_subjectivity', 'all_statements'
            ]
            
            # Calculate percentages
            id_level_metrics['personalized_statement_percentage'] = (
                id_level_metrics['personalized_statements_count'] / 
                id_level_metrics['total_statements'] * 100
            )
            
            id_level_metrics['personalized_word_percentage'] = (
                id_level_metrics['personalized_keywords_count'] / 
                id_level_metrics['total_words'].replace(0, 1) * 100
            )
            
            # Binary flag for having any personalized content
            id_level_metrics['has_personalized_content'] = (
                id_level_metrics['personalized_statements_count'] > 0
            ).astype(int)
        
        return {
            'ground_truth': ground_truth_df,
            'statement_metrics': statement_metrics,
            'id_level_metrics': id_level_metrics,
            'ig_posts': ig_posts_df
        }
        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None

# Main processing
if st.sidebar.button("🔄 Process Data", type="primary"):
    st.session_state.processed_data = process_data()

# Display results if data is processed
if st.session_state.processed_data:
    data = st.session_state.processed_data
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "📝 Statement Level", "🏷️ ID Level", 
        "📈 Visualizations", "💾 Export Data"
    ])
    
    with tab1:
        st.header("📊 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Statements", 
                len(data['statement_metrics']),
                help="Total number of statements analyzed"
            )
        
        with col2:
            personalized_count = len(data['statement_metrics'][data['statement_metrics']['Mode'] == 1])
            st.metric(
                "Personalized Statements", 
                personalized_count,
                delta=f"{personalized_count/len(data['statement_metrics'])*100:.1f}%"
            )
        
        with col3:
            unique_ids = data['statement_metrics']['ID'].nunique()
            st.metric(
                "Unique Posts", 
                unique_ids,
                help="Number of unique Instagram posts"
            )
        
        with col4:
            avg_strength = data['statement_metrics']['personalization_strength'].mean()
            st.metric(
                "Avg Personalization Strength", 
                f"{avg_strength:.3f}",
                help="Average continuous personalization score"
            )
        
        # Distribution charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                data['statement_metrics'], 
                x='personalization_strength',
                title="Distribution of Personalization Strength",
                nbins=20,
                color_discrete_sequence=['#1f77b4']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            sentiment_counts = data['statement_metrics']['sentiment_category'].value_counts()
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Sentiment Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("📝 Statement-Level Analysis")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_strength = st.slider(
                "Min Personalization Strength",
                0.0, 1.0, 0.0, 0.1,
                help="Filter statements by minimum personalization strength"
            )
        
        with col2:
            sentiment_filter = st.selectbox(
                "Sentiment Filter",
                ['All', 'positive', 'negative', 'neutral'],
                help="Filter by sentiment category"
            )
        
        with col3:
            show_keywords = st.checkbox("Show Keywords", True)
        
        # Apply filters
        filtered_statements = data['statement_metrics'][
            data['statement_metrics']['personalization_strength'] >= min_strength
        ]
        
        if sentiment_filter != 'All':
            filtered_statements = filtered_statements[
                filtered_statements['sentiment_category'] == sentiment_filter
            ]
        
        # Display filtered results
        st.subheader(f"Filtered Statements ({len(filtered_statements)} results)")
        
        display_columns = [
            'ID', 'Turn', 'Statement', 'Mode', 'personalization_strength',
            'sentiment_polarity', 'sentiment_category', 'word_count'
        ]
        
        if show_keywords:
            display_columns.append('personalized_keywords')
        
        st.dataframe(
            filtered_statements[display_columns],
            use_container_width=True,
            height=400
        )
        
        # Top keywords
        if show_keywords:
            st.subheader("🔑 Most Common Personalization Keywords")
            all_keywords = []
            for keywords_list in filtered_statements['personalized_keywords']:
                all_keywords.extend(keywords_list)
            
            if all_keywords:
                keyword_counts = Counter(all_keywords)
                top_keywords = keyword_counts.most_common(10)
                
                fig = px.bar(
                    x=[kw[1] for kw in top_keywords],
                    y=[kw[0] for kw in top_keywords],
                    orientation='h',
                    title="Top 10 Personalization Keywords",
                    labels={'x': 'Frequency', 'y': 'Keywords'}
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("🏷️ ID-Level Aggregated Metrics")
        
        # Summary statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Continuous Measures Summary")
            summary_stats = data['id_level_metrics'][
                ['avg_personalization_strength', 'personalized_statement_percentage', 
                 'personalized_word_percentage', 'avg_sentiment_polarity']
            ].describe()
            st.dataframe(summary_stats)
        
        with col2:
            st.subheader("🎯 Binary vs Continuous Comparison")
            binary_vs_continuous = pd.DataFrame({
                'Metric': [
                    'Posts with Personalized Content (Binary)',
                    'Avg Personalization Strength (Continuous)',
                    'Avg Statement Percentage (Continuous)', 
                    'Avg Word Percentage (Continuous)'
                ],
                'Value': [
                    f"{data['id_level_metrics']['has_personalized_content'].sum()} / {len(data['id_level_metrics'])}",
                    f"{data['id_level_metrics']['avg_personalization_strength'].mean():.3f}",
                    f"{data['id_level_metrics']['personalized_statement_percentage'].mean():.1f}%",
                    f"{data['id_level_metrics']['personalized_word_percentage'].mean():.1f}%"
                ]
            })
            st.dataframe(binary_vs_continuous, hide_index=True)
        
        # Display ID-level data
        st.subheader("📋 ID-Level Data")
        
        display_cols = [
            'ID', 'total_statements', 'personalized_statements_count',
            'personalized_statement_percentage', 'personalized_word_percentage',
            'avg_personalization_strength', 'avg_sentiment_polarity',
            'has_personalized_content'
        ]
        
        st.dataframe(
            data['id_level_metrics'][display_cols],
            use_container_width=True,
            height=400
        )
    
    with tab4:
        st.header("📈 Advanced Visualizations")
        
        # Correlation matrix
        st.subheader("🔗 Correlation Analysis")
        
        numeric_cols = [
            'personalized_statements_count', 'total_statements', 'total_words',
            'personalized_keywords_count', 'avg_personalization_strength',
            'personalized_statement_percentage', 'personalized_word_percentage',
            'avg_sentiment_polarity'
        ]
        
        corr_matrix = data['id_level_metrics'][numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            title="Correlation Matrix of Continuous Measures",
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plots
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(
                data['id_level_metrics'],
                x='personalized_statement_percentage',
                y='avg_personalization_strength',
                title='Statement % vs Avg Strength',
                trendline='ols'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                data['id_level_metrics'],
                x='personalized_word_percentage',
                y='avg_sentiment_polarity',
                title='Word % vs Sentiment',
                trendline='ols'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Distribution comparisons
        st.subheader("📊 Distribution Comparisons")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Personalization Strength Distribution',
                'Statement Percentage Distribution',
                'Word Percentage Distribution',
                'Sentiment Distribution'
            ]
        )
        
        # Add histograms
        fig.add_trace(
            go.Histogram(
                x=data['id_level_metrics']['avg_personalization_strength'],
                name='Strength'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(
                x=data['id_level_metrics']['personalized_statement_percentage'],
                name='Statement %'
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Histogram(
                x=data['id_level_metrics']['personalized_word_percentage'],
                name='Word %'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Histogram(
                x=data['id_level_metrics']['avg_sentiment_polarity'],
                name='Sentiment'
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.header("💾 Export Processed Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📤 Generated Outputs")
            
            outputs_info = [
                {
                    "name": "📊 ID-Level Aggregated Metrics",
                    "rows": len(data['id_level_metrics']),
                    "cols": len(data['id_level_metrics'].columns),
                    "description": "Post-level continuous measures"
                },
                {
                    "name": "📝 Statement-Level Word Metrics", 
                    "rows": len(data['statement_metrics']),
                    "cols": len(data['statement_metrics'].columns),
                    "description": "Statement-level analysis"
                },
                {
                    "name": "🔍 Enhanced Ground Truth",
                    "rows": len(data['ground_truth']),
                    "cols": len(data['ground_truth'].columns), 
                    "description": "Enriched original data"
                }
            ]
            
            for output in outputs_info:
                st.write(f"**{output['name']}**")
                st.write(f"• {output['rows']} rows, {output['cols']} columns")
                st.write(f"• {output['description']}")
                st.write("")
        
        with col2:
            st.subheader("⬇️ Download Generated Outputs")
            
            # Primary outputs to generate
            output_datasets = {
                "📊 ID-Level Aggregated Metrics": {
                    "data": data['id_level_metrics'],
                    "filename": "id_level_aggregated_metrics.csv",
                    "description": "Post-level continuous measures and percentages"
                },
                "📝 Statement-Level Word Metrics": {
                    "data": data['statement_metrics'],
                    "filename": "statement_level_word_metrics.csv", 
                    "description": "Statement-level analysis with continuous scores"
                },
                "🔍 Enhanced Ground Truth": {
                    "data": data['ground_truth'],
                    "filename": "enhanced_ground_truth_with_metrics.csv",
                    "description": "Original data enriched with continuous measures"
                }
            }
            
            for name, info in output_datasets.items():
                with st.expander(f"{name} ({len(info['data'])} rows)"):
                    st.write(info['description'])
                    st.write(f"**Columns**: {', '.join(info['data'].columns[:8])}{'...' if len(info['data'].columns) > 8 else ''}")
                    
                    csv_data = info['data'].to_csv(index=False)
                    st.download_button(
                        label=f"📥 Download {name}",
                        data=csv_data,
                        file_name=info['filename'],
                        mime='text/csv',
                        use_container_width=True
                    )
        
        # Summary report
        st.subheader("📋 Analysis Summary")
        
        summary_report = f"""
        ## Classifier Transformation Summary
        
        ### Binary to Continuous Conversion Results:
        
        **1. Personalization Strength Score**
        - Range: 0.0 to 1.0
        - Average: {data['id_level_metrics']['avg_personalization_strength'].mean():.3f}
        - Based on: Binary classification + keyword density
        
        **2. Statement-Level Percentage**
        - Average: {data['id_level_metrics']['personalized_statement_percentage'].mean():.1f}%
        - Range: {data['id_level_metrics']['personalized_statement_percentage'].min():.1f}% to {data['id_level_metrics']['personalized_statement_percentage'].max():.1f}%
        - Measures: Personalized statements / Total statements per post
        
        **3. Word-Level Percentage** 
        - Average: {data['id_level_metrics']['personalized_word_percentage'].mean():.1f}%
        - Range: {data['id_level_metrics']['personalized_word_percentage'].min():.1f}% to {data['id_level_metrics']['personalized_word_percentage'].max():.1f}%
        - Measures: Personalized keywords / Total words per post
        
        ### Dataset Statistics:
        - Total Statements Analyzed: {len(data['statement_metrics'])}
        - Unique Posts: {data['statement_metrics']['ID'].nunique()}
        - Personalized Statements: {len(data['statement_metrics'][data['statement_metrics']['Mode'] == 1])}
        - Posts with Personalized Content: {data['id_level_metrics']['has_personalized_content'].sum()}
        """
        
        st.markdown(summary_report)

else:
    # Instructions when no data is loaded
    st.info("👆 Upload your data files using the sidebar to get started!")
    
    st.markdown("""
    ## 🚀 How to Use This Tool
    
    ### 1. **Upload Required File**
    - **Ground Truth Data**: Contains statements and binary classifications
    - **IG Posts Data** (optional): Original Instagram post data for additional context
    
    ### 2. **Generated Outputs**
    The tool will create these files for you:
    - **📊 id_level_aggregated_metrics.csv**: Post-level continuous measures
    - **📝 statement_level_word_metrics.csv**: Statement-level analysis with scores
    - **🔍 Enhanced ground truth**: Original data enriched with new metrics
    
    ### 3. **Continuous Measures Generated**
    - **Personalization Strength**: 0-1 score combining binary classification + keyword density
    - **Statement Percentage**: % of personalized statements per post
    - **Word Percentage**: % of personalized words per post
    
    ### 4. **Analysis Features**
    - **Statement-level analysis** with keyword extraction
    - **ID-level aggregation** with multiple continuous metrics
    - **Sentiment analysis** integration
    - **Interactive visualizations** and correlations
    - **Export capabilities** for processed data
    
    ### 5. **Key Benefits**
    ✅ Transform binary (0/1) classifications into nuanced continuous measures  
    ✅ Analyze personalization at both statement and post levels  
    ✅ Extract and analyze personalization keywords automatically  
    ✅ Integrate sentiment analysis for richer insights  
    ✅ Export enhanced datasets for further analysis  
    """)

# Footer
st.markdown("---")
st.markdown("**📊 Classifier Word Metrics Tool** - Transform binary classifications into continuous measures")
