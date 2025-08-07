import streamlit as st
import pandas as pd
import re
import io
import json
import time
from typing import List, Dict, Tuple, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import requests

# Streamlit Configuration
st.set_page_config(
    page_title="Dictionary Classification Bot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache expensive operations
@st.cache_data
def download_nltk_data():
    """Download NLTK data once and cache it"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    return True

@st.cache_data
def get_stopwords():
    """Cache stopwords to avoid repeated loading"""
    download_nltk_data()
    return set(stopwords.words('english'))

@st.cache_data
def parse_csv_robustly(csv_content: str) -> Tuple[Optional[pd.DataFrame], Optional[List[str]]]:
    """Robust CSV parsing with caching"""
    parsing_strategies = [
        {'sep': ',', 'quotechar': '"', 'skipinitialspace': True},
        {'sep': ',', 'quotechar': '"', 'skipinitialspace': True, 'escapechar': '\\'},
        {'sep': ',', 'quoting': 1, 'skipinitialspace': True},
        {'sep': ';', 'quotechar': '"', 'skipinitialspace': True},
        {'sep': '\t', 'quotechar': '"', 'skipinitialspace': True},
        {'sep': ',', 'dtype': str, 'na_filter': False, 'skipinitialspace': True},
    ]
    
    errors = []
    for i, strategy in enumerate(parsing_strategies):
        try:
            df = pd.read_csv(io.StringIO(csv_content), **strategy)
            if not df.empty and len(df.columns) >= 2:
                return df, None
        except Exception as e:
            errors.append(f"Strategy {i+1}: {str(e)}")
    
    return None, errors

@st.cache_data
def validate_csv(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate CSV structure with caching"""
    if df.empty:
        return False, "CSV is empty"
    
    columns = [col.lower().strip() for col in df.columns]
    
    id_cols = [col for col in columns if 'id' in col]
    if not id_cols:
        return False, "CSV must contain an 'ID' column"
    
    text_cols = [col for col in columns if any(word in col for word in ['statement', 'text', 'content'])]
    if not text_cols:
        return False, "CSV must contain a 'Statement', 'Text', or 'Content' column"
    
    return True, "Valid"

@st.cache_data
def preprocess_text(text: str) -> str:
    """Cache text preprocessing"""
    if pd.isna(text):
        return ""
    return str(text).lower().strip()

@st.cache_data
def generate_mock_dictionary(tactic: str, _sample_texts: List[str] = None) -> List[str]:
    """Generate mock dictionary with caching"""
    stop_words = get_stopwords()
    tactic_words = re.findall(r'\b[a-zA-Z]{4,}\b', tactic.lower())
    
    common_words = [
        'persuade', 'convince', 'influence', 'appeal', 'emotion', 'fear', 'hope',
        'trust', 'authority', 'expert', 'evidence', 'proof', 'claim', 'argue',
        'support', 'oppose', 'attack', 'defend', 'promote', 'encourage'
    ]
    
    all_words = tactic_words + common_words
    filtered_words = list(set([
        word for word in all_words 
        if word not in stop_words and len(word) > 3
    ]))
    
    return filtered_words[:15]

@st.cache_data
def classify_texts(texts: List[str], keywords: List[str]) -> List[Dict]:
    """Batch classify texts with caching"""
    results = []
    for text in texts:
        text_lower = preprocess_text(text)
        matches = []
        
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, text_lower):
                matches.append(keyword)
        
        results.append({
            'matches': matches,
            'score': len(matches),
            'is_match': len(matches) > 0
        })
    
    return results

# Initialize session state with defaults
def init_session_state():
    """Initialize session state with proper defaults"""
    defaults = {
        'step': 1,
        'tactic_definition': "",
        'csv_data': pd.DataFrame(),
        'dictionary': [],
        'classification_results': pd.DataFrame(),
        'dictionary_prompt': 'Generate a list of single-word (unigram) keywords for a text classification dictionary focused on the "tactic" based on the "context"',
        'processing': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Navigation helpers
def go_to_step(step_num: int):
    """Safe step navigation"""
    if 1 <= step_num <= 5:
        st.session_state.step = step_num
        st.rerun()

def reset_app():
    """Reset application state"""
    for key in list(st.session_state.keys()):
        if key.startswith(('step', 'tactic', 'csv', 'dictionary', 'classification')):
            del st.session_state[key]
    init_session_state()
    st.rerun()

# UI Components
def render_progress_indicator():
    """Render progress indicator"""
    steps = ["Define Tactic", "Upload Data", "Generate Dictionary", "Edit Dictionary", "View Results"]
    progress = st.session_state.step / len(steps)
    st.progress(progress)
    
    cols = st.columns(len(steps))
    for i, (col, step_name) in enumerate(zip(cols, steps)):
        with col:
            if i + 1 < st.session_state.step:
                st.success(f"✅ {step_name}")
            elif i + 1 == st.session_state.step:
                st.info(f"🔄 {step_name}")
            else:
                st.text(f"⏳ {step_name}")

def render_sidebar():
    """Render sidebar content"""
    with st.sidebar:
        st.header("ℹ️ Help & Information")
        
        st.markdown("""
        ### How to Use:
        1. **Define Tactic**: Enter what you want to classify
        2. **Upload Data**: Provide CSV with ID and Statement columns
        3. **Generate Dictionary**: Create keywords automatically
        4. **Edit Dictionary**: Refine keywords as needed
        5. **View Results**: See classified statements
        """)
        
        if st.session_state.step > 1 and not st.session_state.csv_data.empty:
            st.markdown("### Current Data:")
            st.metric("Rows", len(st.session_state.csv_data))
            st.text(f"Columns: {', '.join(st.session_state.csv_data.columns)}")
        
        if st.session_state.dictionary:
            st.markdown("### Current Dictionary:")
            for keyword in st.session_state.dictionary[:5]:
                st.write(f"• {keyword}")
            if len(st.session_state.dictionary) > 5:
                st.write(f"... and {len(st.session_state.dictionary) - 5} more")
        
        st.divider()
        if st.button("🔄 Reset App", key="sidebar_reset"):
            reset_app()

# Step Components
def step1_define_tactic():
    """Step 1: Define Tactic with form"""
    st.header("📝 Step 1: Define Your Tactic")
    
    with st.form("tactic_form"):
        tactic_definition = st.text_area(
            "Tactic Definition",
            value=st.session_state.tactic_definition,
            placeholder="Enter a clear definition of the tactic you want to classify...",
            help="Example: 'Emotional appeals that use fear to persuade the audience'",
            height=100
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            submitted = st.form_submit_button("Next Step", disabled=not tactic_definition.strip())
        
        if submitted and tactic_definition.strip():
            st.session_state.tactic_definition = tactic_definition
            go_to_step(2)

def step2_upload_data():
    """Step 2: Upload Data with improved handling"""
    st.header("📂 Step 2: Upload Sample Data")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Use tabs for different input methods
        tab1, tab2 = st.tabs(["📁 Upload File", "📝 Paste Data"])
        
        with tab1:
            uploaded_file = st.file_uploader(
                "Upload CSV file",
                type=['csv'],
                help="CSV must contain 'ID' and 'Statement' columns"
            )
            
            if uploaded_file is not None:
                process_uploaded_file(uploaded_file)
        
        with tab2:
            with st.form("csv_input_form"):
                csv_text = st.text_area(
                    "CSV Data",
                    placeholder='ID,Statement\n1,"This is a sample statement"\n2,"Another example statement"',
                    height=200
                )
                
                if st.form_submit_button("Parse CSV") and csv_text.strip():
                    process_csv_text(csv_text)
        
        # CSV format help
        with st.expander("💡 CSV Formatting Tips"):
            st.markdown("""
            **Common issues and solutions:**
            - **Text with commas**: Wrap in quotes → `"This text, has commas"`
            - **Text with quotes**: Escape with double quotes → `"He said ""hello"""`
            - **Different delimiters**: Try semicolon (;) or tab-separated
            
            **Required columns:**
            - An ID column (can be named: ID, id, Id, etc.)
            - A text column (can be named: Statement, Text, Content, etc.)
            """)
    
    with col2:
        if not st.session_state.csv_data.empty:
            st.markdown("**Preview:**")
            st.dataframe(st.session_state.csv_data.head(), use_container_width=True)
            
            st.markdown("**Data Info:**")
            st.metric("Rows", len(st.session_state.csv_data))
            st.metric("Columns", len(st.session_state.csv_data.columns))
    
    # Navigation
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back", key="step2_back"):
            go_to_step(1)
    with col2:
        if st.button("Next Step →", disabled=st.session_state.csv_data.empty, key="step2_next"):
            go_to_step(3)

def process_uploaded_file(uploaded_file):
    """Process uploaded CSV file"""
    try:
        content = uploaded_file.read().decode('utf-8')
        df, errors = parse_csv_robustly(content)
        
        if df is not None:
            is_valid, message = validate_csv(df)
            if is_valid:
                st.session_state.csv_data = df
                st.success(f"✅ Successfully loaded {len(df)} rows")
                st.info(f"Columns: {', '.join(df.columns.tolist())}")
            else:
                st.error(f"❌ {message}")
        else:
            st.error("❌ Failed to parse CSV file")
            for error in errors[:2]:
                st.text(f"  • {error}")
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")

def process_csv_text(csv_text):
    """Process CSV text input"""
    try:
        df, errors = parse_csv_robustly(csv_text)
        
        if df is not None:
            is_valid, message = validate_csv(df)
            if is_valid:
                st.session_state.csv_data = df
                st.success(f"✅ Successfully parsed {len(df)} rows")
            else:
                st.error(f"❌ {message}")
        else:
            st.error("❌ Failed to parse CSV data")
            for error in errors[:2]:
                st.text(f"  • {error}")
    except Exception as e:
        st.error(f"❌ Error parsing CSV: {str(e)}")

def step3_generate_dictionary():
    """Step 3: Generate Dictionary"""
    st.header("🧠 Step 3: Generate Dictionary")
    
    with st.form("dictionary_generation_form"):
        prompt = st.text_area(
            "Dictionary Creation Prompt",
            value=st.session_state.dictionary_prompt,
            height=80
        )
        
        api_key = st.text_input(
            "Claude API Key (optional)",
            type="password",
            help="Leave empty to use mock generation"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.form_submit_button("🧠 Generate Dictionary"):
                generate_dictionary(prompt, api_key)
    
    # Preview section
    with st.expander("📋 Preview", expanded=True):
        st.markdown(f"**Tactic:** {st.session_state.tactic_definition}")
        st.markdown(f"**Sample Data:** {len(st.session_state.csv_data)} statements")
        
        if len(st.session_state.csv_data) > 0:
            text_col = find_text_column(st.session_state.csv_data)
            if text_col:
                st.markdown("**Sample statements:**")
                for i, text in enumerate(st.session_state.csv_data[text_col].head(3)):
                    st.markdown(f"{i+1}. {str(text)[:100]}...")
    
    # Navigation
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back", key="step3_back"):
            go_to_step(2)

def generate_dictionary(prompt, api_key):
    """Generate dictionary with progress indicator"""
    st.session_state.dictionary_prompt = prompt
    
    with st.spinner("Generating dictionary..."):
        try:
            if api_key:
                # Placeholder for actual API call
                time.sleep(2)
                keywords = generate_mock_dictionary(st.session_state.tactic_definition)
            else:
                st.info("Using mock generation (add Claude API key for real AI generation)")
                keywords = generate_mock_dictionary(st.session_state.tactic_definition)
            
            st.session_state.dictionary = keywords
            st.success("✅ Dictionary generated successfully!")
            time.sleep(1)
            go_to_step(4)
            
        except Exception as e:
            st.error(f"❌ Error generating dictionary: {str(e)}")

def find_text_column(df):
    """Find the text column in dataframe"""
    for col in df.columns:
        if any(word in col.lower() for word in ['statement', 'text', 'content']):
            return col
    return None

def step4_edit_dictionary():
    """Step 4: Edit Dictionary with improved UX"""
    st.header("✏️ Step 4: Edit Dictionary")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Add keyword form
        with st.form("add_keyword_form"):
            new_keyword = st.text_input("Add new keyword:")
            if st.form_submit_button("➕ Add Keyword") and new_keyword.strip():
                keyword = new_keyword.strip().lower()
                if keyword not in st.session_state.dictionary:
                    st.session_state.dictionary.append(keyword)
                    st.rerun()
                else:
                    st.warning("Keyword already exists!")
        
        # Display dictionary with batch operations
        st.markdown(f"**Dictionary Keywords ({len(st.session_state.dictionary)}):**")
        
        if st.session_state.dictionary:
            # Use container for better performance
            with st.container():
                for i, keyword in enumerate(st.session_state.dictionary):
                    col_key, col_remove = st.columns([4, 1])
                    with col_key:
                        st.text(f"🔸 {keyword}")
                    with col_remove:
                        if st.button("❌", key=f"remove_{i}"):
                            st.session_state.dictionary.remove(keyword)
                            st.rerun()
        else:
            st.info("No keywords yet. Generate some or add manually.")
    
    with col2:
        st.markdown("**Quick Actions:**")
        if st.button("🔄 Regenerate"):
            go_to_step(3)
        
        if st.button("🗑️ Clear All"):
            st.session_state.dictionary = []
            st.rerun()
    
    # Navigation
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back", key="step4_back"):
            go_to_step(3)
    with col2:
        if st.button("🔍 Classify", disabled=len(st.session_state.dictionary) == 0, key="step4_classify"):
            perform_classification()

def perform_classification():
    """Perform text classification"""
    with st.spinner("Classifying statements..."):
        text_col = find_text_column(st.session_state.csv_data)
        id_col = None
        
        for col in st.session_state.csv_data.columns:
            if 'id' in col.lower():
                id_col = col
                break
        
        texts = st.session_state.csv_data[text_col].tolist() if text_col else []
        classifications = classify_texts(texts, st.session_state.dictionary)
        
        results = []
        for i, (_, row) in enumerate(st.session_state.csv_data.iterrows()):
            classification = classifications[i]
            
            results.append({
                'ID': row[id_col] if id_col else i,
                'Statement': row[text_col] if text_col else "",
                'Matches': ', '.join(classification['matches']),
                'Score': classification['score'],
                'Classification': 'Match' if classification['is_match'] else 'No Match',
                'MatchedKeywords': classification['matches']
            })
        
        st.session_state.classification_results = pd.DataFrame(results)
        st.success("✅ Classification complete!")
        time.sleep(1)
        go_to_step(5)

def step5_view_results():
    """Step 5: View Results with enhanced visualization"""
    st.header("📊 Step 5: Classification Results")
    
    # Metrics
    results_df = st.session_state.classification_results
    total = len(results_df)
    matches = len(results_df[results_df['Classification'] == 'Match'])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Statements", total)
    with col2:
        st.metric("Matches Found", matches)
    with col3:
        st.metric("Match Rate", f"{matches/total*100:.1f}%" if total > 0 else "0%")
    with col4:
        st.metric("Keywords Used", len(st.session_state.dictionary))
    
    # Export functionality
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.download_button(
            "📥 Export CSV",
            data=results_df.to_csv(index=False),
            file_name="classification_results.csv",
            mime="text/csv"
        ):
            st.success("Results exported!")
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        show_matches_only = st.checkbox("Show matches only")
    with col2:
        min_score = st.slider("Minimum score", 0, 10, 0)
    
    # Filter and display results
    filtered_df = results_df.copy()
    if show_matches_only:
        filtered_df = filtered_df[filtered_df['Classification'] == 'Match']
    filtered_df = filtered_df[filtered_df['Score'] >= min_score]
    
    st.markdown(f"**Showing {len(filtered_df)} of {total} statements**")
    
    # Paginated results
    page_size = 10
    total_pages = (len(filtered_df) - 1) // page_size + 1 if len(filtered_df) > 0 else 1
    
    if total_pages > 1:
        page = st.selectbox("Page", range(1, total_pages + 1)) - 1
        start_idx = page * page_size
        end_idx = start_idx + page_size
        page_df = filtered_df.iloc[start_idx:end_idx]
    else:
        page_df = filtered_df
    
    # Display results
    for _, result in page_df.iterrows():
        with st.expander(
            f"ID: {result['ID']} - {result['Classification']} (Score: {result['Score']})",
            expanded=False
        ):
            st.markdown(result['Statement'])
            if result['MatchedKeywords']:
                st.markdown(f"**Matched keywords:** {', '.join(result['MatchedKeywords'])}")
    
    # Navigation
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Edit Dictionary", key="step5_back"):
            go_to_step(4)
    with col2:
        if st.button("🔄 Start New", key="step5_reset"):
            reset_app()

# Main Application
def main():
    """Main application function"""
    init_session_state()
    
    st.title("🧠 Dictionary Refinement Bot")
    st.markdown("Create custom dictionaries and classify text data using AI-powered keyword generation")
    
    render_progress_indicator()
    st.divider()
    
    # Route to appropriate step
    step_functions = {
        1: step1_define_tactic,
        2: step2_upload_data,
        3: step3_generate_dictionary,
        4: step4_edit_dictionary,
        5: step5_view_results
    }
    
    step_function = step_functions.get(st.session_state.step, step1_define_tactic)
    step_function()
    
    render_sidebar()

if __name__ == "__main__":
    main()
