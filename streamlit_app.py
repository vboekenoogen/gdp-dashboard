import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import re
import random
from collections import Counter
import hashlib
import time

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
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize session state with proper defaults
def initialize_session_state():
    """Initialize session state variables"""
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'csv_data' not in st.session_state:
        st.session_state.csv_data = None
    if 'keywords' not in st.session_state:
        st.session_state.keywords = []
    if 'data_hash' not in st.session_state:
        st.session_state.data_hash = None
    if 'keywords_hash' not in st.session_state:
        st.session_state.keywords_hash = None

# Utility functions
@st.cache_data
def get_file_hash(file_contents):
    """Generate hash for file contents to detect changes"""
    return hashlib.md5(file_contents).hexdigest()

@st.cache_data
def get_keywords_hash(keywords):
    """Generate hash for keywords to detect changes"""
    return hashlib.md5(''.join(sorted(keywords)).encode()).hexdigest()

# Mock LLM rating function
@st.cache_data
def rate_with_llm(statement):
    """Mock LLM implementation - returns random score between 1-5"""
    # Use statement as seed for consistent results
    random.seed(hash(statement) % 2**32)
    
    personalized_words = ['custom', 'personal', 'tailored', 'individual', 'unique', 'specific']
    lower_statement = statement.lower()
    
    base_score = 1
    for word in personalized_words:
        if word in lower_statement:
            base_score += random.random() * 0.8 + 0.2
    
    return min(5, max(1, base_score + (random.random() - 0.5) * 0.5))

# Optimized file processing functions
@st.cache_data
def detect_encoding(file_contents):
    """Detect file encoding using chardet if available, otherwise try common encodings"""
    try:
        import chardet
        result = chardet.detect(file_contents[:10000])  # Sample first 10KB
        confidence = result.get('confidence', 0)
        detected_encoding = result.get('encoding', 'utf-8')
        
        if confidence > 0.7:
            return [detected_encoding, 'utf-8', 'latin-1', 'windows-1252']
        else:
            return ['utf-8', detected_encoding, 'latin-1', 'windows-1252']
    except ImportError:
        return ['utf-8', 'latin-1', 'windows-1252', 'cp1252', 'iso-8859-1']

@st.cache_data
def process_csv_content(file_contents, filename, _encodings_to_try):
    """Process CSV content with caching for performance"""
    for encoding in _encodings_to_try:
        try:
            content = file_contents.decode(encoding)
            df = pd.read_csv(StringIO(content))
            return df, encoding, None
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            if "codec can't decode" not in str(e).lower():
                return None, encoding, str(e)
    
    return None, None, "Could not decode with any encoding"

def process_csv_files(uploaded_files):
    """Process uploaded CSV files with optimization"""
    if not uploaded_files:
        return None
    
    all_data = []
    
    for file in uploaded_files:
        file_contents = file.read()
        file_hash = get_file_hash(file_contents)
        
        # Check if we've already processed this file
        cache_key = f"csv_{file.name}_{file_hash}"
        
        encodings_to_try = detect_encoding(file_contents)
        df, encoding, error = process_csv_content(file_contents, file.name, encodings_to_try)
        
        if df is not None:
            all_data.append(df)
            st.success(f"✅ Successfully read {file.name} with {encoding} encoding")
        else:
            st.error(f"❌ Could not read {file.name}: {error}")
            st.info("💡 Try saving the file as UTF-8 CSV format.")
            return None
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    return None

@st.cache_data
def process_keyword_content(file_contents, filename, _encodings_to_try):
    """Process keyword file content with caching"""
    for encoding in _encodings_to_try:
        try:
            if filename.endswith('.csv'):
                content = file_contents.decode(encoding)
                df = pd.read_csv(StringIO(content), header=None)
                keywords = df.values.flatten().tolist()
            else:
                content = file_contents.decode(encoding)
                keywords = re.split(r'[,\n\r]+', content)
            
            # Clean keywords
            keywords = [word.strip() for word in keywords if word.strip()]
            return keywords, encoding, None
            
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            if "codec can't decode" not in str(e).lower():
                return None, encoding, str(e)
    
    return None, None, "Could not decode with any encoding"

def process_keyword_file(uploaded_file):
    """Process uploaded keyword file with optimization"""
    if not uploaded_file:
        return []
    
    file_contents = uploaded_file.read()
    encodings_to_try = detect_encoding(file_contents)
    
    keywords, encoding, error = process_keyword_content(
        file_contents, uploaded_file.name, encodings_to_try
    )
    
    if keywords:
        st.success(f"✅ Successfully read keyword file with {encoding} encoding")
        return keywords
    else:
        st.error(f"❌ Could not read keyword file: {error}")
        return []

# Optimized column detection
@st.cache_data
def find_flexible_column(column_list, possible_names):
    """Find column with flexible name matching - cached for performance"""
    for name in possible_names:
        for col in column_list:
            if name.lower() in col.lower():
                return col
    return None

# Main data processing function with caching
@st.cache_data
def process_data_cached(csv_data_dict, keywords_list, debug_mode=False):
    """Cached version of data processing for performance"""
    # Convert dict back to DataFrame (for caching compatibility)
    csv_data = pd.DataFrame(csv_data_dict)
    keywords = keywords_list
    
    if csv_data is None or len(keywords) == 0:
        return None, "No data or keywords provided"
    
    debug_output = ""
    
    try:
        # Find columns flexibly
        column_list = list(csv_data.columns)
        statement_col = find_flexible_column(column_list, ['statement', 'text', 'content'])
        post_id_col = find_flexible_column(column_list, ['post_id', 'post id', 'id', 'postid'])
        word_count_col = find_flexible_column(column_list, ['word_count', 'word count', 'wordcount'])
        
        if debug_mode:
            debug_output += f"Available columns: {column_list}\n"
            debug_output += f"Found statement column: {statement_col}\n"
            debug_output += f"Found post_id column: {post_id_col}\n"
            debug_output += f"Found word_count column: {word_count_col}\n\n"
        
        if not statement_col or not post_id_col:
            error_msg = f"Required columns not found.\n"
            error_msg += f"Available columns: {column_list}\n"
            error_msg += f"Looking for statement column in: ['statement', 'text', 'content']\n"
            error_msg += f"Looking for post_id column in: ['post_id', 'post id', 'id', 'postid']\n"
            error_msg += f"Found statement: {statement_col}, Found post_id: {post_id_col}"
            return None, error_msg
        
        # Use vectorized operations for better performance
        valid_mask = (
            csv_data[statement_col].notna() & 
            csv_data[post_id_col].notna() & 
            (csv_data[statement_col].astype(str).str.strip() != '') &
            (csv_data[post_id_col].astype(str).str.strip() != '')
        )
        
        valid_data = csv_data[valid_mask].copy()
        
        if len(valid_data) == 0:
            return None, "No valid rows found after filtering. Check that your data has non-empty statement and post_id columns."
        
        if debug_mode:
            debug_output += f"Keywords ({len(keywords)}): {', '.join(keywords)}\n"
            debug_output += f"CSV Data Rows: {len(csv_data)}\n"
            debug_output += f"Valid rows after filtering: {len(valid_data)}\n"
            debug_output += f"Using columns - Statement: {statement_col}, Post ID: {post_id_col}\n\n"
        
        # Vectorized keyword matching for performance
        keywords_lower = [kw.lower() for kw in keywords]
        valid_data['statement_lower'] = valid_data[statement_col].astype(str).str.lower()
        
        # Create binary match column
        def check_keywords(text):
            return any(kw in text for kw in keywords_lower)
        
        valid_data['has_match'] = valid_data['statement_lower'].apply(check_keywords)
        
        # Calculate post-level statistics efficiently
        post_stats = valid_data.groupby(post_id_col).agg({
            'has_match': ['count', 'sum']
        }).round(2)
        
        post_stats.columns = ['total', 'matches']
        post_stats['match_pct'] = (post_stats['matches'] / post_stats['total']) * 100
        post_stats_dict = post_stats.to_dict('index')
        
        if debug_mode:
            total_matches = valid_data['has_match'].sum()
            debug_output += f"Total matches found: {total_matches}/{len(valid_data)}\n"
            debug_output += f"Match rate: {(total_matches/len(valid_data)*100):.1f}%\n\n"
        
        # Process each row efficiently
        processed_rows = []
        
        for _, row in valid_data.iterrows():
            statement = str(row[statement_col])
            post_id = str(row[post_id_col])
            statement_lower = row['statement_lower']
            
            # Efficient word processing
            words = re.findall(r'\b\w+\b', statement_lower)
            word_count = row[word_count_col] if word_count_col and pd.notna(row[word_count_col]) else len(words)
            
            # Binary classification
            binary_match = 1 if row['has_match'] else 0
            
            # Dictionary word percentage
            matching_words = [word for word in words if any(
                word == kw or kw in word or word in kw 
                for kw in keywords_lower
            )]
            
            dict_word_pct = (len(matching_words) / len(words)) * 100 if words else 0
            
            # Post match percentage
            post_match_pct = post_stats_dict[post_id]['match_pct']
            
            # Mock LLM score (cached)
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
            debug_output += f"Final Results:\n"
            debug_output += f"Binary matches: {binary_matches}/{len(processed_rows)}\n"
            
            non_zero_dict = [row for row in processed_rows if row['Dict Word %'] > 0]
            debug_output += f"Non-zero dict_word_pct: {len(non_zero_dict)}\n"
            if non_zero_dict:
                avg_dict = sum(row['Dict Word %'] for row in non_zero_dict) / len(non_zero_dict)
                debug_output += f"Average dict_word_pct (non-zero): {avg_dict:.2f}%\n"
        
        return pd.DataFrame(processed_rows), debug_output
        
    except Exception as e:
        error_msg = f"Error processing data: {str(e)}\n"
        error_msg += f"Available columns: {list(csv_data.columns) if csv_data is not None else 'None'}\n"
        error_msg += f"Data shape: {csv_data.shape if csv_data is not None else 'None'}"
        return None, error_msg

# Non-cached wrapper for UI
def process_data(csv_data, keywords, debug_mode=False):
    """Wrapper function that converts DataFrame to dict for caching"""
    if csv_data is None:
        return None, "No data provided"
    
    # Convert DataFrame to dict for caching compatibility
    csv_data_dict = csv_data.to_dict('records')
    return process_data_cached(csv_data_dict, keywords, debug_mode)

# Optimized visualization functions
@st.cache_data
def create_histogram_data(dict_word_pcts):
    """Create histogram data with caching"""
    bins = 10
    max_val = max(dict_word_pcts) if dict_word_pcts else 1
    bin_size = max_val / bins
    bin_counts = [0] * bins
    bin_labels = []

    for i in range(bins):
        bin_labels.append(f"{(i * bin_size):.1f}-{((i + 1) * bin_size):.1f}%")

    for pct in dict_word_pcts:
        bin_index = min(int(pct / bin_size), bins - 1)
        bin_counts[bin_index] += 1

    return bin_labels, bin_counts

@st.cache_data
def calculate_classifier_strength(processed_data_dict):
    """Calculate classifier strength metrics with caching"""
    df = pd.DataFrame(processed_data_dict)
    
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
    
    return avg_statement_rate, avg_word_rate

# Header
st.markdown("""
<div class="main-header">
    <h1>💖 Classifier Word Metrics Tool</h1>
    <p>Analyze text data with keyword-based classification and advanced metrics</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
initialize_session_state()

# Sidebar for file uploads and keywords
with st.sidebar:
    st.header("📁 Data Input")
    
    # CSV file upload
    st.subheader("CSV Data Files")
    csv_files = st.file_uploader(
        "Upload CSV files with: statement, post_id, word_count (optional)",
        type=['csv'],
        accept_multiple_files=True,
        key="csv_upload",
        help="Supports multiple encodings (UTF-8, Latin-1, Windows-1252)"
    )
    
    if csv_files:
        with st.spinner("Reading CSV files..."):
            new_csv_data = process_csv_files(csv_files)
            if new_csv_data is not None:
                # Check if data has changed
                new_hash = get_file_hash(str(new_csv_data.values.tobytes()))
                if st.session_state.data_hash != new_hash:
                    st.session_state.csv_data = new_csv_data
                    st.session_state.data_hash = new_hash
                    st.session_state.processed_data = None  # Reset processed data
                
                st.success(f"✅ Successfully loaded {len(st.session_state.csv_data)} rows from {len(csv_files)} file(s)")
                
                # Show column information in expander
                with st.expander("📋 Data Preview", expanded=False):
                    st.write("**Available columns:**")
                    cols = list(st.session_state.csv_data.columns)
                    for i, col in enumerate(cols):
                        st.write(f"{i+1}. `{col}`")
                    
                    st.write("**Sample data:**")
                    st.dataframe(st.session_state.csv_data.head(3), use_container_width=True)
            else:
                st.error("❌ Failed to load CSV files. Please check the file encoding and format.")
    
    # Keywords input
    st.subheader("Keywords Input")
    
    # Keyword file upload
    keyword_file = st.file_uploader(
        "Upload keyword file (.csv/.txt)",
        type=['csv', 'txt'],
        key="keyword_upload",
        help="One keyword per line or comma-separated"
    )
    
    if keyword_file:
        uploaded_keywords = process_keyword_file(keyword_file)
        if uploaded_keywords:
            new_keywords_hash = get_keywords_hash(uploaded_keywords)
            if st.session_state.keywords_hash != new_keywords_hash:
                st.session_state.keywords = uploaded_keywords
                st.session_state.keywords_hash = new_keywords_hash
                st.session_state.processed_data = None  # Reset processed data
            st.success(f"✅ Loaded {len(uploaded_keywords)} keywords")
    
    st.write("**— OR —**")
    
    # Manual keyword input
    default_keywords = "custom, tailored, personalized, bespoke, individualized, customized, personal, unique, specialized, exclusive, made-to-order, one-of-a-kind"
    
    keywords_text = st.text_area(
        "Enter keywords (comma or line separated):",
        value=default_keywords,
        height=150,
        key="keywords_input",
        help="Enter one keyword per line or separate with commas"
    )
    
    if keywords_text and not keyword_file:
        manual_keywords = [kw.strip() for kw in re.split(r'[,\n\r]+', keywords_text) if kw.strip()]
        new_keywords_hash = get_keywords_hash(manual_keywords)
        if st.session_state.keywords_hash != new_keywords_hash:
            st.session_state.keywords = manual_keywords
            st.session_state.keywords_hash = new_keywords_hash
            st.session_state.processed_data = None  # Reset processed data
    
    # Debug mode
    debug_mode = st.checkbox("Enable Debug Mode", help="Shows detailed matching information")
    
    # Performance settings
    with st.expander("⚙️ Performance Settings"):
        st.write("**Memory Usage:**")
        if st.session_state.csv_data is not None:
            memory_usage = st.session_state.csv_data.memory_usage(deep=True).sum() / 1024**2
            st.write(f"CSV Data: {memory_usage:.1f} MB")
        
        if st.button("🗑️ Clear Cache", help="Clear all cached data to free memory"):
            st.cache_data.clear()
            st.success("Cache cleared!")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🚀 Process Data")
    
    # Process button
    process_button = st.button(
        "🚀 Process Data & Generate Metrics", 
        type="primary", 
        use_container_width=True,
        disabled=(st.session_state.csv_data is None or not st.session_state.keywords)
    )
    
    if process_button:
        if st.session_state.csv_data is None:
            st.error("Please upload CSV files first!")
        elif not st.session_state.keywords:
            st.error("Please enter keywords or upload a keyword file!")
        else:
            # Check if we need to reprocess
            current_data_hash = get_file_hash(str(st.session_state.csv_data.values.tobytes()))
            current_keywords_hash = get_keywords_hash(st.session_state.keywords)
            
            need_reprocess = (
                st.session_state.processed_data is None or
                st.session_state.data_hash != current_data_hash or
                st.session_state.keywords_hash != current_keywords_hash
            )
            
            if need_reprocess:
                with st.spinner("Processing your data..."):
                    start_time = time.time()
                    
                    try:
                        processed_data, debug_output = process_data(
                            st.session_state.csv_data, 
                            st.session_state.keywords, 
                            debug_mode
                        )
                        
                        processing_time = time.time() - start_time
                        
                        if processed_data is not None:
                            st.session_state.processed_data = processed_data
                            st.session_state.data_hash = current_data_hash
                            st.session_state.keywords_hash = current_keywords_hash
                            
                            st.success(f"✅ Data processed successfully in {processing_time:.2f}s!")
                            
                            if debug_mode and debug_output:
                                st.subheader("🔍 Debug Information")
                                st.text(debug_output)
                        else:
                            st.error("❌ Processing failed.")
                            if debug_output:
                                st.error(debug_output)
                                
                    except Exception as e:
                        st.error(f"❌ An error occurred during processing: {str(e)}")
                        st.info("💡 Please check your data format and try again. Enable debug mode for more details.")
            else:
                st.info("✅ Using cached results (data unchanged)")

with col2:
    if st.session_state.csv_data is not None:
        st.subheader("📊 Data Info")
        st.metric("Rows", len(st.session_state.csv_data))
        st.metric("Columns", len(st.session_state.csv_data.columns))
        
        # Memory usage
        memory_mb = st.session_state.csv_data.memory_usage(deep=True).sum() / 1024**2
        st.metric("Memory", f"{memory_mb:.1f} MB")
        
    if st.session_state.keywords:
        st.subheader("🔤 Keywords")
        st.metric("Count", len(st.session_state.keywords))
        with st.expander("View Keywords"):
            st.write(", ".join(st.session_state.keywords[:10]))
            if len(st.session_state.keywords) > 10:
                st.write(f"... and {len(st.session_state.keywords) - 10} more")

# Results section
if st.session_state.processed_data is not None:
    st.header("📊 Results")
    
    df = st.session_state.processed_data
    
    # Statistics overview with better layout
    metrics_container = st.container()
    with metrics_container:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Statements", len(df))
        with col2:
            total_matches = len(df[df['Binary Match'] == 1])
            st.metric("Keyword Matches", total_matches)
        with col3:
            avg_dict_pct = df['Dict Word %'].mean()
            st.metric("Avg Dict Word %", f"{avg_dict_pct:.1f}%")
        with col4:
            avg_llm_score = df['LLM Score'].mean()
            st.metric("Avg LLM Score", f"{avg_llm_score:.1f}")
        with col5:
            unique_posts = df['Post ID'].nunique()
            st.metric("Unique Posts", unique_posts)
    
    # Data table with enhanced filtering
    st.subheader("📋 Results Data")
    
    # Advanced filtering options
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        filter_text = st.text_input("🔍 Search all columns:", key="filter_input")
    with filter_col2:
        match_filter = st.selectbox("Filter by matches:", ["All", "Matches Only", "No Matches"])
    
    # Apply filters
    filtered_df = df.copy()
    
    if filter_text:
        mask = df.astype(str).apply(lambda x: x.str.contains(filter_text, case=False, na=False)).any(axis=1)
        filtered_df = filtered_df[mask]
    
    if match_filter == "Matches Only":
        filtered_df = filtered_df[filtered_df['Binary Match'] == 1]
    elif match_filter == "No Matches":
        filtered_df = filtered_df[filtered_df['Binary Match'] == 0]
    
    # Display table with pagination
    st.write(f"Showing {len(filtered_df)} of {len(df)} rows")
    
    # Use column configuration for better display
    column_config = {
        "Statement": st.column_config.TextColumn(
            "Statement",
            width="large",
            help="Original text statement"
        ),
        "Binary Match": st.column_config.NumberColumn(
            "Binary Match",
            format="%d",
            help="1 if contains keywords, 0 otherwise"
        ),
        "Dict Word %": st.column_config.NumberColumn(
            "Dict Word %",
            format="%.2f%%",
            help="Percentage of words matching keywords"
        ),
        "Post Match %": st.column_config.NumberColumn(
            "Post Match %",
            format="%.2f%%",
            help="Percentage of statements in post with matches"
        ),
        "LLM Score": st.column_config.NumberColumn(
            "LLM Score",
            format="%.2f",
            help="Mock AI confidence score (1-5)"
        )
    }
    
    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=400,
        column_config=column_config
    )
    
    # Download options with better organization
    download_col1, download_col2, download_col3 = st.columns(3)
    
    with download_col1:
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="💾 Download CSV",
            data=csv_data,
            file_name=f"classifier_metrics_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with download_col2:
        excel_data = df.to_csv(index=False, sep='\t')
        st.download_button(
            label="📋 Download TSV (Excel)",
            data=excel_data,
            file_name=f"classifier_metrics_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.tsv",
            mime="text/tab-separated-values",
            use_container_width=True
        )
    
    with download_col3:
        json_data = df.to_json(orient='records', indent=2)
        st.download_button(
            label="📄 Download JSON",
            data=json_data,
            file_name=f"classifier_metrics_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    # Visualizations with better performance
    st.subheader("📊 Visualizations")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Histogram with cached data
        dict_word_pcts = df['Dict Word %'].tolist()
        bin_labels, bin_counts = create_histogram_data(dict_word_pcts)
        
        fig_hist = px.bar(
            x=bin_labels,
            y=bin_counts,
            title="Distribution of Dictionary Word Percentage",
            labels={'x': 'Dictionary Word Percentage', 'y': 'Number of Statements'},
            color_discrete_sequence=['#ff69b4']
        )
        fig_hist.update_layout(showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with viz_col2:
        # Classifier Strength Analysis with cached calculations
        st.subheader("🎯 Classifier Strength")
        
        processed_data_dict = df.to_dict('records')
        avg_statement_rate, avg_word_rate = calculate_classifier_strength(processed_data_dict)
        
        def get_strength_label(rate):
            if rate >= 50:
                return "High", "#38a169"
            elif rate >= 20:
                return "Medium", "#ed8936"
            else:
                return "Low", "#e53e3e"
        
        statement_strength, statement_color = get_strength_label(avg_statement_rate)
        word_strength, word_color = get_strength_label(avg_word_rate)
        
        # Use metrics for better display
        st.metric(
            label="Statement-Level Rate",
            value=f"{avg_statement_rate:.1f}%",
            help="Percentage of statements classified as positive per post"
        )
        st.markdown(f"<span style='color: {statement_color}; font-weight: bold;'>{statement_strength} Strength</span>", unsafe_allow_html=True)
        
        st.metric(
            label="Word-Level Rate", 
            value=f"{avg_word_rate:.1f}%",
            help="Percentage of classifier words out of total words per post"
        )
        st.markdown(f"<span style='color: {word_color}; font-weight: bold;'>{word_strength} Strength</span>", unsafe_allow_html=True)
        
        # Interpretation guide
        with st.expander("💡 Strength Interpretation"):
            st.write("**High (>50%):** Strong classifier presence")
            st.write("**Medium (20-50%):** Moderate classifier presence")
            st.write("**Low (<20%):** Weak classifier presence")

# Footer with performance info
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("Built with ❤️ using Streamlit")

with footer_col2:
    if st.session_state.processed_data is not None:
        cache_info = st.cache_data.get_stats()
        st.write(f"Cache hits: {len(cache_info)}")

with footer_col3:
    if st.button("🔄 Refresh App"):
        st.experimental_rerun()
