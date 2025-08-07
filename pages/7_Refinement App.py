import streamlit as st
import pandas as pd
import numpy as np
import re
import json
from typing import List, Dict, Tuple
from io import StringIO
import base64

# Set page config
st.set_page_config(
    page_title="Dictionary Classification Bot",
    page_icon="🧠",
    layout="wide"
)

def init_session_state():
    """Initialize session state variables"""
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'tactic_definition' not in st.session_state:
        st.session_state.tactic_definition = ""
    if 'csv_data' not in st.session_state:
        st.session_state.csv_data = None
    if 'dictionary' not in st.session_state:
        st.session_state.dictionary = []
    if 'classification_results' not in st.session_state:
        st.session_state.classification_results = None
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None

def parse_csv(uploaded_file) -> pd.DataFrame:
    """Parse uploaded CSV file and validate columns"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Check for required columns (case insensitive)
        required_cols = ['id', 'statement', 'true_label']
        df_cols_lower = [col.lower() for col in df.columns]
        
        # Map columns
        column_mapping = {}
        for req_col in required_cols:
            found_col = None
            for i, col in enumerate(df_cols_lower):
                if req_col in col or col in req_col:
                    found_col = df.columns[i]
                    break
            if found_col is None:
                raise ValueError(f"Required column '{req_col}' not found. Available columns: {list(df.columns)}")
            column_mapping[req_col] = found_col
        
        # Rename columns for consistency
        df = df.rename(columns={
            column_mapping['id']: 'id',
            column_mapping['statement']: 'statement', 
            column_mapping['true_label']: 'true_label'
        })
        
        # Clean data
        df = df[['id', 'statement', 'true_label']].dropna()
        df['statement'] = df['statement'].astype(str)
        df['true_label'] = df['true_label'].astype(str).str.lower().str.strip()
        
        # Convert true labels to binary (1 for positive class, 0 for negative)
        positive_values = ['1', 'true', 'yes', 'match', 'positive']
        df['true_label_binary'] = df['true_label'].apply(
            lambda x: 1 if x in positive_values else 0
        )
        
        return df
        
    except Exception as e:
        st.error(f"Error parsing CSV: {str(e)}")
        return None

def generate_dictionary_mock(tactic_definition: str, sample_statements: List[str]) -> List[str]:
    """
    Mock dictionary generation function.
    In a real implementation, this would call an AI API like Claude or OpenAI.
    """
    # This is a mock implementation - replace with actual AI API call
    # For demonstration, generate some sample keywords based on common terms
    
    sample_keywords = []
    
    # Extract keywords from tactic definition
    tactic_words = re.findall(r'\b[a-zA-Z]{3,}\b', tactic_definition.lower())
    sample_keywords.extend(tactic_words[:5])
    
    # Extract frequent words from sample statements
    all_text = ' '.join(sample_statements).lower()
    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text)
    word_freq = {}
    for word in words:
        if len(word) > 3 and word not in ['this', 'that', 'with', 'from', 'they', 'have', 'been', 'were']:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Get top frequent words
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    sample_keywords.extend([word for word, freq in top_words])
    
    # Remove duplicates and return
    return list(set(sample_keywords))

def classify_statements(df: pd.DataFrame, dictionary: List[str]) -> pd.DataFrame:
    """Classify statements using dictionary keywords"""
    results = []
    
    for _, row in df.iterrows():
        statement = row['statement'].lower()
        
        # Find matching keywords
        matches = []
        for keyword in dictionary:
            # Use word boundary regex for exact word matching
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, statement):
                matches.append(keyword)
        
        # Calculate prediction (1 if any matches, 0 if no matches)
        prediction = 1 if len(matches) > 0 else 0
        score = len(matches)
        
        results.append({
            'id': row['id'],
            'statement': row['statement'],
            'true_label': row['true_label'],
            'true_label_binary': row['true_label_binary'],
            'prediction': prediction,
            'matches': matches,
            'score': score
        })
    
    return pd.DataFrame(results)

def calculate_metrics(results_df: pd.DataFrame) -> Dict:
    """Calculate precision, recall, F1, and confusion matrix"""
    y_true = results_df['true_label_binary'].values
    y_pred = results_df['prediction'].values
    
    # Calculate confusion matrix components
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'total_samples': len(results_df),
        'positive_predictions': np.sum(y_pred),
        'true_positives_count': np.sum(y_true)
    }

def highlight_keywords(text: str, keywords: List[str]) -> str:
    """Highlight keywords in text for display"""
    highlighted = text
    for keyword in keywords:
        pattern = r'\b' + re.escape(keyword) + r'\b'
        highlighted = re.sub(pattern, f'**{keyword}**', highlighted, flags=re.IGNORECASE)
    return highlighted

def create_download_link(df: pd.DataFrame, filename: str) -> str:
    """Create download link for DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def main():
    init_session_state()
    
    # Header
    st.title("🧠 Dictionary Classification Bot")
    st.markdown("Create custom dictionaries and classify text data with precision, recall, and F1 metrics")
    
    # Progress indicator
    steps = ["Define Tactic", "Upload Data", "Generate Dictionary", "Edit Dictionary", "View Results"]
    
    cols = st.columns(5)
    for i, step_name in enumerate(steps, 1):
        with cols[i-1]:
            if st.session_state.step >= i:
                st.success(f"✅ {step_name}")
            else:
                st.info(f"⏳ {step_name}")
    
    st.divider()
    
    # Step 1: Define Tactic
    if st.session_state.step == 1:
        st.header("📝 Step 1: Define Your Tactic")
        
        tactic_def = st.text_area(
            "Tactic Definition",
            value=st.session_state.tactic_definition,
            placeholder="Enter a clear definition of the tactic you want to classify...",
            height=120,
            help="Provide a detailed description of what you're trying to identify in the text"
        )
        
        if st.button("Next Step", disabled=not tactic_def.strip()):
            st.session_state.tactic_definition = tactic_def
            st.session_state.step = 2
            st.rerun()
    
    # Step 2: Upload Data
    elif st.session_state.step == 2:
        st.header("📤 Step 2: Upload Sample Data")
        
        st.info("Upload a CSV file with columns: **ID**, **Statement**, **True_Label**")
        st.markdown("""
        **Required columns:**
        - **ID**: Unique identifier for each statement
        - **Statement**: Text to be classified
        - **True_Label**: Ground truth labels (1/0, true/false, match/no_match, etc.)
        """)
        
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        
        if uploaded_file is not None:
            df = parse_csv(uploaded_file)
            if df is not None:
                st.session_state.csv_data = df
                
                st.success(f"✅ Successfully parsed {len(df)} statements")
                
                # Show preview
                st.subheader("Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Show label distribution
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Statements", len(df))
                with col2:
                    positive_count = df['true_label_binary'].sum()
                    st.metric("Positive Labels", f"{positive_count} ({positive_count/len(df)*100:.1f}%)")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back"):
                st.session_state.step = 1
                st.rerun()
        with col2:
            if st.button("Next Step →", disabled=st.session_state.csv_data is None):
                st.session_state.step = 3
                st.rerun()
    
    # Step 3: Generate Dictionary
    elif st.session_state.step == 3:
        st.header("🧠 Step 3: Generate Dictionary")
        
        st.markdown(f"**Tactic:** {st.session_state.tactic_definition}")
        st.markdown(f"**Sample Data:** {len(st.session_state.csv_data)} statements ready for analysis")
        
        dictionary_prompt = st.text_area(
            "Dictionary Generation Instructions",
            value="Generate a list of single-word (unigram) keywords for text classification focused on the tactic based on the context",
            height=100
        )
        
        if st.button("🎯 Generate Dictionary"):
            with st.spinner("Generating dictionary keywords..."):
                # Get sample statements for dictionary generation
                sample_statements = st.session_state.csv_data['statement'].head(10).tolist()
                
                # Generate dictionary (mock implementation)
                generated_keywords = generate_dictionary_mock(
                    st.session_state.tactic_definition, 
                    sample_statements
                )
                
                st.session_state.dictionary = generated_keywords
                st.session_state.step = 4
                st.rerun()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back"):
                st.session_state.step = 2
                st.rerun()
    
    # Step 4: Edit Dictionary
    elif st.session_state.step == 4:
        st.header("✏️ Step 4: Edit Dictionary")
        
        # Add new keyword
        col1, col2 = st.columns([3, 1])
        with col1:
            new_keyword = st.text_input("Add new keyword", placeholder="Enter keyword...")
        with col2:
            if st.button("➕ Add") and new_keyword.strip():
                if new_keyword.lower() not in [k.lower() for k in st.session_state.dictionary]:
                    st.session_state.dictionary.append(new_keyword.lower())
                    st.rerun()
        
        # Display current dictionary
        st.subheader(f"Dictionary Keywords ({len(st.session_state.dictionary)})")
        
        if st.session_state.dictionary:
            # Create a grid of keywords with delete buttons
            cols_per_row = 4
            for i in range(0, len(st.session_state.dictionary), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, keyword in enumerate(st.session_state.dictionary[i:i+cols_per_row]):
                    with cols[j]:
                        col_inner1, col_inner2 = st.columns([3, 1])
                        with col_inner1:
                            st.code(keyword)
                        with col_inner2:
                            if st.button("🗑️", key=f"del_{keyword}"):
                                st.session_state.dictionary.remove(keyword)
                                st.rerun()
        else:
            st.info("No keywords in dictionary yet. Add some keywords above.")
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back"):
                st.session_state.step = 3
                st.rerun()
        with col2:
            if st.button("🎯 Classify Statements", disabled=len(st.session_state.dictionary) == 0):
                with st.spinner("Classifying statements..."):
                    results = classify_statements(st.session_state.csv_data, st.session_state.dictionary)
                    metrics = calculate_metrics(results)
                    
                    st.session_state.classification_results = results
                    st.session_state.metrics = metrics
                    st.session_state.step = 5
                    st.rerun()
    
    # Step 5: View Results
    elif st.session_state.step == 5:
        st.header("📊 Step 5: Classification Results")
        
        results_df = st.session_state.classification_results
        metrics = st.session_state.metrics
        
        # Metrics Dashboard
        st.subheader("📈 Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Precision", f"{metrics['precision']:.3f}")
        with col2:
            st.metric("Recall", f"{metrics['recall']:.3f}")
        with col3:
            st.metric("F1 Score", f"{metrics['f1']:.3f}")
        with col4:
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        
        # Confusion Matrix
        st.subheader("🔢 Confusion Matrix")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("True Positives", metrics['tp'])
        with col2:
            st.metric("False Positives", metrics['fp'])
        with col3:
            st.metric("False Negatives", metrics['fn'])
        with col4:
            st.metric("True Negatives", metrics['tn'])
        
        # Summary Statistics
        st.subheader("📋 Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Statements", metrics['total_samples'])
        with col2:
            st.metric("Predicted Positive", metrics['positive_predictions'])
        with col3:
            st.metric("Actually Positive", metrics['true_positives_count'])
        
        # Detailed Results
        st.subheader("🔍 Detailed Results")
        
        # Filter options
        filter_option = st.selectbox(
            "Filter results",
            ["All", "Correct Predictions", "Incorrect Predictions", "True Positives", "False Positives", "False Negatives", "True Negatives"]
        )
        
        # Apply filter
        filtered_df = results_df.copy()
        if filter_option == "Correct Predictions":
            filtered_df = filtered_df[filtered_df['true_label_binary'] == filtered_df['prediction']]
        elif filter_option == "Incorrect Predictions":
            filtered_df = filtered_df[filtered_df['true_label_binary'] != filtered_df['prediction']]
        elif filter_option == "True Positives":
            filtered_df = filtered_df[(filtered_df['true_label_binary'] == 1) & (filtered_df['prediction'] == 1)]
        elif filter_option == "False Positives":
            filtered_df = filtered_df[(filtered_df['true_label_binary'] == 0) & (filtered_df['prediction'] == 1)]
        elif filter_option == "False Negatives":
            filtered_df = filtered_df[(filtered_df['true_label_binary'] == 1) & (filtered_df['prediction'] == 0)]
        elif filter_option == "True Negatives":
            filtered_df = filtered_df[(filtered_df['true_label_binary'] == 0) & (filtered_df['prediction'] == 0)]
        
        st.info(f"Showing {len(filtered_df)} of {len(results_df)} results")
        
        # Display results
        for _, row in filtered_df.iterrows():
            # Determine card color based on prediction correctness
            is_correct = row['true_label_binary'] == row['prediction']
            
            if is_correct:
                if row['prediction'] == 1:
                    card_color = "🟢"  # True Positive
                else:
                    card_color = "🔵"  # True Negative
            else:
                if row['prediction'] == 1:
                    card_color = "🟡"  # False Positive
                else:
                    card_color = "🔴"  # False Negative
            
            with st.container():
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.markdown(f"**{card_color} ID: {row['id']}**")
                    # Highlight matched keywords in statement
                    if row['matches']:
                        highlighted_text = highlight_keywords(row['statement'], row['matches'])
                        st.markdown(highlighted_text)
                    else:
                        st.text(row['statement'])
                    
                    if row['matches']:
                        st.caption(f"Matched keywords: {', '.join(row['matches'])}")
                
                with col2:
                    st.metric("True Label", row['true_label'])
                    st.metric("Prediction", "Match" if row['prediction'] == 1 else "No Match")
                    st.metric("Score", row['score'])
                
                st.divider()
        
        # Export Results
        st.subheader("💾 Export Results")
        
        # Prepare export data
        export_df = results_df.copy()
        export_df['matches_str'] = export_df['matches'].apply(lambda x: '; '.join(x))
        export_df['prediction_label'] = export_df['prediction'].apply(lambda x: "Match" if x == 1 else "No Match")
        
        export_cols = ['id', 'statement', 'true_label', 'prediction_label', 'matches_str', 'score']
        export_df = export_df[export_cols]
        export_df.columns = ['ID', 'Statement', 'True_Label', 'Prediction', 'Matched_Keywords', 'Score']
        
        # Download button
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results CSV",
            data=csv,
            file_name="classification_results.csv",
            mime="text/csv"
        )
        
        # Navigation
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Edit Dictionary"):
                st.session_state.step = 4
                st.rerun()
        with col2:
            if st.button("🔄 Start New Classification"):
                # Reset all session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

if __name__ == "__main__":
    main()
