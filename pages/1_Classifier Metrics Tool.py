import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import re
import random
from collections import Counter

# Set page config
st.set_page_config(
    page_title="💖 Classifier Word Metrics Tool",
    page_icon="💖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #ff69b4 0%, #ff1493 50%, #dc143c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ff69b4, #ff1493);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .metric-number {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    .stSelectbox > div > div > div {
        background-color: #fdf2f8;
    }
    
    .stTextArea > div > div > textarea {
        background-color: #fdf2f8;
    }
    
    .success-box {
        background-color: #fdf2f8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff69b4;
        margin: 1rem 0;
    }
    
    .debug-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        font-family: monospace;
        font-size: 12px;
        max-height: 300px;
        overflow-y: auto;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>💖 Classifier Word Metrics Tool</h1>
    <p>Analyze text data with keyword-based classification and advanced metrics</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'csv_data' not in st.session_state:
    st.session_state.csv_data = None
if 'keywords' not in st.session_state:
    st.session_state.keywords = []

# Mock LLM rating function
def rate_with_llm(statement):
    """Mock LLM implementation - returns random score between 1-5"""
    personalized_words = ['custom', 'personal', 'tailored', 'individual', 'unique', 'specific']
    lower_statement = statement.lower()
    
    base_score = 1
    for word in personalized_words:
        if word in lower_statement:
            base_score += random.random() * 0.8 + 0.2
    
    return min(5, max(1, base_score + (random.random() - 0.5) * 0.5))

# File processing functions
def process_csv_files(uploaded_files):
    """Process uploaded CSV files"""
    all_data = []
    
    for file in uploaded_files:
        try:
            df = pd.read_csv(file)
            all_data.append(df)
        except Exception as e:
            st.error(f"Error reading {file.name}: {str(e)}")
            return None
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    return None

def process_keyword_file(uploaded_file):
    """Process uploaded keyword file"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, header=None)
            keywords = df.values.flatten().tolist()
        else:
            content = uploaded_file.read().decode('utf-8')
            keywords = re.split(r'[,\n\r]+', content)
        
        # Clean keywords
        keywords = [word.strip() for word in keywords if word.strip()]
        return keywords
    except Exception as e:
        st.error(f"Error reading keyword file: {str(e)}")
        return []

def find_flexible_column(df, possible_names):
    """Find column with flexible name matching"""
    df_columns = df.columns.str.lower()
    for name in possible_names:
        matches = df_columns[df_columns.str.contains(name.lower(), na=False)]
        if len(matches) > 0:
            return df.columns[df_columns == matches.iloc[0]].iloc[0]
    return None

def process_data(csv_data, keywords, debug_mode=False):
    """Process the data with keyword matching"""
    if csv_data is None or len(keywords) == 0:
        return None, ""
    
    debug_output = ""
    
    # Find columns flexibly
    statement_col = find_flexible_column(csv_data, ['statement', 'text', 'content'])
    post_id_col = find_flexible_column(csv_data, ['post_id', 'post id', 'id', 'postid'])
    word_count_col = find_flexible_column(csv_data, ['word_count', 'word count', 'wordcount'])
    
    if not statement_col or not post_id_col:
        available_cols = list(csv_data.columns)
        return None, f"Required columns not found. Available columns: {available_cols}"
    
    if debug_mode:
        debug_output += f"Keywords ({len(keywords)}): {', '.join(keywords)}\n"
        debug_output += f"CSV Data Rows: {len(csv_data)}\n"
        debug_output += f"Using columns - Statement: {statement_col}, Post ID: {post_id_col}\n\n"
    
    # Filter valid rows
    valid_data = csv_data.dropna(subset=[statement_col, post_id_col])
    valid_data = valid_data[valid_data[statement_col].astype(str).str.strip() != '']
    valid_data = valid_data[valid_data[post_id_col].astype(str).str.strip() != '']
    
    if debug_mode:
        debug_output += f"Valid rows after filtering: {len(valid_data)}\n"
        if len(valid_data) > 0:
            debug_output += f"Sample statements:\n"
            for i, row in valid_data.head(3).iterrows():
                debug_output += f"{i+1}. \"{row[statement_col]}\" (post_id: {row[post_id_col]})\n"
            debug_output += "\n"
    
    # Calculate post-level statistics
    post_stats = {}
    total_processed = 0
    total_matches = 0
    
    for _, row in valid_data.iterrows():
        statement = str(row[statement_col])
        post_id = str(row[post_id_col])
        
        total_processed += 1
        if post_id not in post_stats:
            post_stats[post_id] = {'total': 0, 'matches': 0}
        post_stats[post_id]['total'] += 1
        
        # Check for keyword matches
        statement_lower = statement.lower()
        has_match = any(keyword.lower() in statement_lower for keyword in keywords)
        
        if has_match:
            post_stats[post_id]['matches'] += 1
            total_matches += 1
            
            if debug_mode and total_matches <= 5:
                matching_keywords = [kw for kw in keywords if kw.lower() in statement_lower]
                debug_output += f"✅ Match found: {matching_keywords} in \"{statement[:50]}...\"\n"
    
    if debug_mode:
        debug_output += f"\nFirst Pass Results:\n"
        debug_output += f"Total rows processed: {total_processed}\n"
        debug_output += f"Total matches found: {total_matches}\n"
        debug_output += f"Match rate: {(total_matches/total_processed*100):.1f}%\n\n"
    
    # Calculate post match percentages
    for post_id in post_stats:
        post_stats[post_id]['match_pct'] = (post_stats[post_id]['matches'] / post_stats[post_id]['total']) * 100
    
    # Process each row with full metrics
    processed_rows = []
    
    for _, row in valid_data.iterrows():
        statement = str(row[statement_col])
        post_id = str(row[post_id_col])
        statement_lower = statement.lower()
        
        # Clean and split words
        words = re.findall(r'\b\w+\b', statement_lower)
        word_count = row[word_count_col] if word_count_col and pd.notna(row[word_count_col]) else len(words)
        
        # Binary classification
        binary_match = 1 if any(keyword.lower() in statement_lower for keyword in keywords) else 0
        
        # Dictionary word percentage
        matching_words = [word for word in words if any(
            word == keyword.lower() or 
            keyword.lower() in word or 
            word in keyword.lower() 
            for keyword in keywords
        )]
        
        dict_word_pct = (len(matching_words) / len(words)) * 100 if words else 0
        
        # Post match percentage
        post_match_pct = post_stats[post_id]['match_pct']
        
        # Mock LLM score
        llm_score = rate_with_llm(statement)
        
        processed_rows.append({
            'Post ID': post_id,
            'Statement': statement,
            'Binary Match': binary_match,
            'Dict Word %': round(dict_word_pct, 2),
            'Post Match %': round(post_match_pct, 2),
            'LLM Score': round(llm_score, 2),
            'Word Count': int(word_count)
        })
    
    if debug_mode:
        binary_matches = sum(1 for row in processed_rows if row['Binary Match'] == 1)
        debug_output += f"Second Pass Results:\n"
        debug_output += f"Binary matches: {binary_matches}/{len(processed_rows)}\n"
        
        non_zero_dict = [row for row in processed_rows if row['Dict Word %'] > 0]
        debug_output += f"Non-zero dict_word_pct: {len(non_zero_dict)}\n"
        if non_zero_dict:
            avg_dict = sum(row['Dict Word %'] for row in non_zero_dict) / len(non_zero_dict)
            debug_output += f"Average dict_word_pct (non-zero): {avg_dict:.2f}%\n"
    
    return pd.DataFrame(processed_rows), debug_output

# Sidebar for file uploads and keywords
with st.sidebar:
    st.header("📁 Data Input")
    
    # CSV file upload
    st.subheader("CSV Data Files")
    csv_files = st.file_uploader(
        "Upload CSV files with: statement, post_id, word_count (optional)",
        type=['csv'],
        accept_multiple_files=True,
        key="csv_upload"
    )
    
    if csv_files:
        st.session_state.csv_data = process_csv_files(csv_files)
        if st.session_state.csv_data is not None:
            st.success(f"✅ Loaded {len(st.session_state.csv_data)} rows from {len(csv_files)} file(s)")
    
    # Keywords input
    st.subheader("Keywords Input")
    
    # Keyword file upload
    keyword_file = st.file_uploader(
        "Upload keyword file (.csv/.txt)",
        type=['csv', 'txt'],
        key="keyword_upload"
    )
    
    if keyword_file:
        uploaded_keywords = process_keyword_file(keyword_file)
        if uploaded_keywords:
            st.session_state.keywords = uploaded_keywords
            st.success(f"✅ Loaded {len(uploaded_keywords)} keywords")
    
    st.write("**— OR —**")
    
    # Manual keyword input
    default_keywords = "custom, tailored, personalized, bespoke, individualized, customized, personal, unique, specialized, exclusive, made-to-order, one-of-a-kind"
    
    keywords_text = st.text_area(
        "Enter keywords (comma or line separated):",
        value=default_keywords,
        height=150,
        key="keywords_input"
    )
    
    if keywords_text and not keyword_file:
        manual_keywords = [kw.strip() for kw in re.split(r'[,\n\r]+', keywords_text) if kw.strip()]
        st.session_state.keywords = manual_keywords
    
    # Debug mode
    debug_mode = st.checkbox("Enable Debug Mode", help="Shows detailed matching information")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🚀 Process Data")
    
    # Process button
    if st.button("🚀 Process Data & Generate Metrics", type="primary", use_container_width=True):
        if st.session_state.csv_data is None:
            st.error("Please upload CSV files first!")
        elif not st.session_state.keywords:
            st.error("Please enter keywords or upload a keyword file!")
        else:
            with st.spinner("Processing your data..."):
                processed_data, debug_output = process_data(
                    st.session_state.csv_data, 
                    st.session_state.keywords, 
                    debug_mode
                )
                
                if processed_data is not None:
                    st.session_state.processed_data = processed_data
                    st.success("✅ Data processed successfully!")
                    
                    if debug_mode and debug_output:
                        st.subheader("🔍 Debug Information")
                        st.markdown(f'<div class="debug-box">{debug_output}</div>', unsafe_allow_html=True)
                else:
                    st.error("❌ Processing failed. Please check your data format.")

with col2:
    if st.session_state.csv_data is not None:
        st.subheader("📊 Data Preview")
        st.write(f"**Rows:** {len(st.session_state.csv_data)}")
        st.write(f"**Columns:** {', '.join(st.session_state.csv_data.columns[:3])}...")
        
    if st.session_state.keywords:
        st.subheader("🔤 Keywords")
        st.write(f"**Count:** {len(st.session_state.keywords)}")
        st.write(f"**Sample:** {', '.join(st.session_state.keywords[:5])}...")

# Results section
if st.session_state.processed_data is not None:
    st.header("📊 Results")
    
    df = st.session_state.processed_data
    
    # Statistics overview
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{len(df)}</div>
            <div class="metric-label">Total Statements</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_matches = len(df[df['Binary Match'] == 1])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{total_matches}</div>
            <div class="metric-label">Keyword Matches</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_dict_pct = df['Dict Word %'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{avg_dict_pct:.1f}%</div>
            <div class="metric-label">Avg Dict Word %</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_llm_score = df['LLM Score'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{avg_llm_score:.1f}</div>
            <div class="metric-label">Avg LLM Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        unique_posts = df['Post ID'].nunique()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{unique_posts}</div>
            <div class="metric-label">Unique Posts</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Data table with filtering
    st.subheader("📋 Results Data")
    
    # Filter input
    filter_text = st.text_input("🔍 Filter results (search any column):", key="filter_input")
    
    # Apply filter
    if filter_text:
        mask = df.astype(str).apply(lambda x: x.str.contains(filter_text, case=False, na=False)).any(axis=1)
        filtered_df = df[mask]
    else:
        filtered_df = df
    
    # Display table
    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=400
    )
    
    # Download options
    col1, col2 = st.columns(2)
    with col1:
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="💾 Download Complete Results CSV",
            data=csv_data,
            file_name=f"classifier_word_metrics_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        excel_data = df.to_csv(index=False, sep='\t')
        st.download_button(
            label="📋 Download Excel Format (TSV)",
            data=excel_data,
            file_name=f"classifier_word_metrics_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.tsv",
            mime="text/tab-separated-values",
            use_container_width=True
        )
    
    # Visualizations
    st.subheader("📊 Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram of Dictionary Word Percentage
        fig_hist = px.histogram(
            df, 
            x='Dict Word %',
            nbins=10,
            title="Distribution of Dictionary Word Percentage",
            color_discrete_sequence=['#ff69b4']
        )
        fig_hist.update_layout(
            xaxis_title="Dictionary Word Percentage",
            yaxis_title="Number of Statements"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Classifier Strength Analysis
        st.subheader("🎯 Classifier Strength Analysis")
        
        # Calculate post-level metrics
        post_metrics = df.groupby('Post ID').agg({
            'Binary Match': ['count', 'sum'],
            'Word Count': 'sum'
        }).round(2)
        
        post_metrics.columns = ['total_statements', 'classified_statements', 'total_words']
        post_metrics['statement_rate'] = (post_metrics['classified_statements'] / post_metrics['total_statements']) * 100
        
        # Calculate word-level classification rate
        classifier_words_per_post = df.groupby('Post ID').apply(
            lambda x: sum((x['Dict Word %'] / 100) * x['Word Count'])
        )
        post_metrics['word_rate'] = (classifier_words_per_post / post_metrics['total_words']) * 100
        
        avg_statement_rate = post_metrics['statement_rate'].mean()
        avg_word_rate = post_metrics['word_rate'].mean()
        
        def get_strength_label(rate):
            if rate >= 50:
                return "High", "#38a169"
            elif rate >= 20:
                return "Medium", "#ed8936"
            else:
                return "Low", "#e53e3e"
        
        statement_strength, statement_color = get_strength_label(avg_statement_rate)
        word_strength, word_color = get_strength_label(avg_word_rate)
        
        st.markdown(f"""
        <div class="success-box">
            <h4 style="color: #ff1493; margin-bottom: 10px;">📊 Statement-Level Classification Rate</h4>
            <div style="font-size: 24px; font-weight: bold; color: {statement_color};">
                {avg_statement_rate:.1f}% ({statement_strength})
            </div>
            <div style="font-size: 12px; color: #666;">Average Classification Rate</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="success-box">
            <h4 style="color: #ff1493; margin-bottom: 10px;">📝 Word-Level Classification Rate</h4>
            <div style="font-size: 24px; font-weight: bold; color: {word_color};">
                {avg_word_rate:.1f}% ({word_strength})
            </div>
            <div style="font-size: 12px; color: #666;">Average Word Classification Rate</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="margin-top: 20px; padding: 15px; background: #fff; border-radius: 8px; border: 1px solid #ff69b4;">
            <h5 style="color: #ff1493; margin-bottom: 10px;">💡 Classifier Strength Interpretation:</h5>
            <div style="font-size: 13px; color: #666;">
                <div><strong>High (>50%):</strong> Strong classifier presence</div>
                <div><strong>Medium (20-50%):</strong> Moderate classifier presence</div>
                <div><strong>Low (<20%):</strong> Weak classifier presence</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")
