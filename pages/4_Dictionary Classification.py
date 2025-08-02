import streamlit as st
import pandas as pd
import re
import io
import json
import time
from typing import List, Dict, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import requests

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# Page configuration
st.set_page_config(
    page_title="Dictionary Classification Bot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'tactic_definition' not in st.session_state:
    st.session_state.tactic_definition = ""
if 'csv_data' not in st.session_state:
    st.session_state.csv_data = pd.DataFrame()
if 'dictionary' not in st.session_state:
    st.session_state.dictionary = []
if 'classification_results' not in st.session_state:
    st.session_state.classification_results = pd.DataFrame()
if 'dictionary_prompt' not in st.session_state:
    st.session_state.dictionary_prompt = 'Generate a list of single-word (unigram) keywords for a text classification dictionary focused on the "tactic" based on the "context"'

def reset_app():
    """Reset the entire application state"""
    st.session_state.step = 1
    st.session_state.tactic_definition = ""
    st.session_state.csv_data = pd.DataFrame()
    st.session_state.dictionary = []
    st.session_state.classification_results = pd.DataFrame()

def validate_csv(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate CSV has required columns"""
    if df.empty:
        return False, "CSV is empty"
    
    columns = [col.lower().strip() for col in df.columns]
    
    # Check for ID column
    id_cols = [col for col in columns if 'id' in col]
    if not id_cols:
        return False, "CSV must contain an 'ID' column"
    
    # Check for statement/text column
    text_cols = [col for col in columns if any(word in col for word in ['statement', 'text', 'content'])]
    if not text_cols:
        return False, "CSV must contain a 'Statement', 'Text', or 'Content' column"
    
    return True, "Valid"

def preprocess_text(text: str) -> str:
    """Clean and preprocess text"""
    if pd.isna(text):
        return ""
    return str(text).lower().strip()

def generate_mock_dictionary(tactic: str, sample_texts: List[str] = None) -> List[str]:
    """Generate a mock dictionary based on tactic definition"""
    # Extract words from tactic definition
    tactic_words = re.findall(r'\b[a-zA-Z]{4,}\b', tactic.lower())
    
    # Common classification keywords
    common_words = [
        'persuade', 'convince', 'influence', 'appeal', 'emotion', 'fear', 'hope',
        'trust', 'authority', 'expert', 'evidence', 'proof', 'claim', 'argue',
        'support', 'oppose', 'attack', 'defend', 'promote', 'encourage'
    ]
    
    # Combine and filter
    stop_words = set(stopwords.words('english'))
    all_words = tactic_words + common_words
    
    # Remove stopwords and duplicates
    filtered_words = list(set([
        word for word in all_words 
        if word not in stop_words and len(word) > 3
    ]))
    
    return filtered_words[:15]  # Return up to 15 keywords

def call_claude_api(prompt: str, api_key: str = None) -> List[str]:
    """Call Claude API for dictionary generation (placeholder for actual implementation)"""
    # This is a placeholder - you would implement actual Claude API call here
    # For now, return mock data
    time.sleep(2)  # Simulate API delay
    return generate_mock_dictionary(st.session_state.tactic_definition)

def classify_text(text: str, keywords: List[str]) -> Dict:
    """Classify a single text using the dictionary"""
    text_lower = preprocess_text(text)
    matches = []
    
    for keyword in keywords:
        # Use word boundaries for more accurate matching
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        if re.search(pattern, text_lower):
            matches.append(keyword)
    
    return {
        'matches': matches,
        'score': len(matches),
        'is_match': len(matches) > 0
    }

def highlight_keywords(text: str, keywords: List[str]) -> str:
    """Highlight keywords in text for display"""
    highlighted = text
    for keyword in keywords:
        pattern = r'\b(' + re.escape(keyword) + r')\b'
        highlighted = re.sub(pattern, r'**\1**', highlighted, flags=re.IGNORECASE)
    return highlighted

def export_results_csv(results_df: pd.DataFrame) -> io.StringIO:
    """Export results to CSV format"""
    output = io.StringIO()
    results_df.to_csv(output, index=False)
    return output.getvalue()

# Main app layout
st.title("🧠 Dictionary Classification Bot")
st.markdown("Create custom dictionaries and classify text data using AI-powered keyword generation")

# Progress indicator
steps = ["Define Tactic", "Upload Data", "Generate Dictionary", "Edit Dictionary", "View Results"]
progress = st.session_state.step / len(steps)
st.progress(progress)

# Step indicator
cols = st.columns(len(steps))
for i, (col, step_name) in enumerate(zip(cols, steps)):
    with col:
        if i + 1 <= st.session_state.step:
            st.success(f"✅ {step_name}")
        elif i + 1 == st.session_state.step:
            st.info(f"🔄 {step_name}")
        else:
            st.text(f"⏳ {step_name}")

st.divider()

# Step 1: Define Tactic
if st.session_state.step == 1:
    st.header("📝 Step 1: Define Your Tactic")
    
    with st.container():
        st.session_state.tactic_definition = st.text_area(
            "Tactic Definition",
            value=st.session_state.tactic_definition,
            placeholder="Enter a clear definition of the tactic you want to classify...",
            help="Example: 'Emotional appeals that use fear to persuade the audience'",
            height=100
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Next Step", disabled=not st.session_state.tactic_definition.strip()):
                st.session_state.step = 2
                st.rerun()

# Step 2: Upload Data
elif st.session_state.step == 2:
    st.header("📂 Step 2: Upload Sample Data")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # File upload option
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="CSV must contain 'ID' and 'Statement' columns"
        )
        
        # Text input option
        st.markdown("**Or paste CSV data directly:**")
        csv_text = st.text_area(
            "CSV Data",
            placeholder="ID,Statement\n1,This is a sample statement\n2,Another example statement",
            height=200
        )
        
        # Process uploaded file or text input
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                is_valid, message = validate_csv(df)
                if is_valid:
                    st.session_state.csv_data = df
                    st.success(f"✅ Successfully loaded {len(df)} rows")
                else:
                    st.error(f"❌ {message}")
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
        
        elif csv_text.strip():
            try:
                df = pd.read_csv(io.StringIO(csv_text))
                is_valid, message = validate_csv(df)
                if is_valid:
                    st.session_state.csv_data = df
                    st.success(f"✅ Successfully parsed {len(df)} rows")
                else:
                    st.error(f"❌ {message}")
            except Exception as e:
                st.error(f"❌ Error parsing CSV: {str(e)}")
    
    with col2:
        if not st.session_state.csv_data.empty:
            st.markdown("**Preview:**")
            st.dataframe(st.session_state.csv_data.head(), use_container_width=True)
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button("← Back"):
            st.session_state.step = 1
            st.rerun()
    with col2:
        if st.button("Next Step →", disabled=st.session_state.csv_data.empty):
            st.session_state.step = 3
            st.rerun()

# Step 3: Generate Dictionary
elif st.session_state.step == 3:
    st.header("🧠 Step 3: Generate Dictionary")
    
    # Prompt customization
    st.session_state.dictionary_prompt = st.text_area(
        "Dictionary Creation Prompt",
        value=st.session_state.dictionary_prompt,
        height=80
    )
    
    # Preview
    with st.expander("Preview", expanded=True):
        st.markdown(f"**Tactic:** {st.session_state.tactic_definition}")
        st.markdown(f"**Sample Data:** {len(st.session_state.csv_data)} statements ready for analysis")
        
        if len(st.session_state.csv_data) > 0:
            # Show sample statements
            text_col = None
            for col in st.session_state.csv_data.columns:
                if any(word in col.lower() for word in ['statement', 'text', 'content']):
                    text_col = col
                    break
            
            if text_col:
                st.markdown("**Sample statements:**")
                for i, text in enumerate(st.session_state.csv_data[text_col].head(3)):
                    st.markdown(f"{i+1}. {text[:100]}...")
    
    # API Key input (optional)
    api_key = st.text_input(
        "Claude API Key (optional)",
        type="password",
        help="Leave empty to use mock generation"
    )
    
    # Generate button
    col1, col2, col3 = st.columns([1, 1, 3])
    
    with col1:
        if st.button("← Back"):
            st.session_state.step = 2
            st.rerun()
    
    with col2:
        if st.button("🧠 Generate Dictionary"):
            with st.spinner("Generating dictionary..."):
                try:
                    if api_key:
                        # Use actual API (implement your Claude API call here)
                        keywords = call_claude_api(st.session_state.dictionary_prompt, api_key)
                    else:
                        # Use mock generation
                        st.info("Using mock generation (add Claude API key for real AI generation)")
                        keywords = generate_mock_dictionary(st.session_state.tactic_definition)
                    
                    st.session_state.dictionary = keywords
                    st.session_state.step = 4
                    st.success("✅ Dictionary generated successfully!")
                    time.sleep(1)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error generating dictionary: {str(e)}")

# Step 4: Edit Dictionary
elif st.session_state.step == 4:
    st.header("✏️ Step 4: Edit Dictionary")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Add new keyword
        new_keyword = st.text_input("Add new keyword:")
        if st.button("➕ Add Keyword") and new_keyword.strip():
            keyword = new_keyword.strip().lower()
            if keyword not in st.session_state.dictionary:
                st.session_state.dictionary.append(keyword)
                st.rerun()
            else:
                st.warning("Keyword already exists!")
        
        # Display current dictionary
        st.markdown(f"**Dictionary Keywords ({len(st.session_state.dictionary)}):**")
        
        if st.session_state.dictionary:
            # Create a grid of keywords with remove buttons
            for i, keyword in enumerate(st.session_state.dictionary):
                col_key, col_remove = st.columns([4, 1])
                with col_key:
                    st.markdown(f"🔸 {keyword}")
                with col_remove:
                    if st.button("❌", key=f"remove_{i}"):
                        st.session_state.dictionary.remove(keyword)
                        st.rerun()
        else:
            st.info("No keywords yet. Generate some or add manually.")
    
    with col2:
        st.markdown("**Quick Actions:**")
        if st.button("🔄 Regenerate Dictionary"):
            st.session_state.step = 3
            st.rerun()
        
        if st.button("🗑️ Clear All"):
            st.session_state.dictionary = []
            st.rerun()
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button("← Back"):
            st.session_state.step = 3
            st.rerun()
    with col2:
        if st.button("🔍 Classify", disabled=len(st.session_state.dictionary) == 0):
            # Perform classification
            with st.spinner("Classifying statements..."):
                results = []
                
                # Find text column
                text_col = None
                id_col = None
                for col in st.session_state.csv_data.columns:
                    if any(word in col.lower() for word in ['statement', 'text', 'content']):
                        text_col = col
                    if 'id' in col.lower():
                        id_col = col
                
                for _, row in st.session_state.csv_data.iterrows():
                    text = row[text_col] if text_col else ""
                    classification = classify_text(text, st.session_state.dictionary)
                    
                    results.append({
                        'ID': row[id_col] if id_col else "",
                        'Statement': text,
                        'Matches': ', '.join(classification['matches']),
                        'Score': classification['score'],
                        'Classification': 'Match' if classification['is_match'] else 'No Match',
                        'MatchedKeywords': classification['matches']
                    })
                
                st.session_state.classification_results = pd.DataFrame(results)
                st.session_state.step = 5
                st.success("✅ Classification complete!")
                time.sleep(1)
                st.rerun()

# Step 5: View Results
elif st.session_state.step == 5:
    st.header("📊 Step 5: Classification Results")
    
    # Summary metrics
    total_statements = len(st.session_state.classification_results)
    matches_found = len(st.session_state.classification_results[
        st.session_state.classification_results['Classification'] == 'Match'
    ])
    keywords_used = len(st.session_state.dictionary)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Statements", total_statements)
    with col2:
        st.metric("Matches Found", matches_found)
    with col3:
        st.metric("Match Rate", f"{matches_found/total_statements*100:.1f}%" if total_statements > 0 else "0%")
    with col4:
        st.metric("Keywords Used", keywords_used)
    
    # Export button
    if st.button("📥 Export Results as CSV"):
        csv_data = export_results_csv(st.session_state.classification_results)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="classification_results.csv",
            mime="text/csv"
        )
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        show_matches_only = st.checkbox("Show matches only")
    with col2:
        min_score = st.slider("Minimum score", 0, 10, 0)
    
    # Filter results
    filtered_results = st.session_state.classification_results.copy()
    if show_matches_only:
        filtered_results = filtered_results[filtered_results['Classification'] == 'Match']
    filtered_results = filtered_results[filtered_results['Score'] >= min_score]
    
    # Display results
    st.markdown(f"**Showing {len(filtered_results)} of {total_statements} statements**")
    
    for _, result in filtered_results.iterrows():
        with st.expander(
            f"ID: {result['ID']} - {result['Classification']} (Score: {result['Score']})",
            expanded=False
        ):
            # Highlight keywords in the statement
            highlighted_text = highlight_keywords(result['Statement'], result['MatchedKeywords'])
            st.markdown(highlighted_text)
            
            if result['MatchedKeywords']:
                st.markdown(f"**Matched keywords:** {', '.join(result['MatchedKeywords'])}")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button("← Edit Dictionary"):
            st.session_state.step = 4
            st.rerun()
    with col2:
        if st.button("🔄 Start New"):
            reset_app()
            st.rerun()

# Sidebar with help and information
with st.sidebar:
    st.header("ℹ️ Help & Information")
    
    st.markdown("""
    ### How to Use:
    1. **Define Tactic**: Enter what you want to classify
    2. **Upload Data**: Provide CSV with ID and Statement columns
    3. **Generate Dictionary**: Create keywords automatically
    4. **Edit Dictionary**: Refine keywords as needed
    5. **View Results**: See classified statements
    
    ### CSV Format:
    Your CSV should have at least these columns:
    - **ID**: Unique identifier for each row
    - **Statement/Text/Content**: The text to classify
    
    ### Example CSV:
    ```
    ID,Statement
    1,This uses fear to persuade people
    2,This is a neutral statement
    ```
    """)
    
    if st.session_state.step > 1 and not st.session_state.csv_data.empty:
        st.markdown("### Current Data:")
        st.write(f"Rows: {len(st.session_state.csv_data)}")
        st.write(f"Columns: {', '.join(st.session_state.csv_data.columns)}")
    
    if st.session_state.dictionary:
        st.markdown("### Current Dictionary:")
        for keyword in st.session_state.dictionary[:5]:
            st.write(f"• {keyword}")
        if len(st.session_state.dictionary) > 5:
            st.write(f"... and {len(st.session_state.dictionary) - 5} more")
    
    st.divider()
    if st.button("🔄 Reset App"):
        reset_app()
        st.rerun()
