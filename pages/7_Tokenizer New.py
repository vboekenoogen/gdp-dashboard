import streamlit as st
import pandas as pd
import nltk
import re
import io
import chardet

# Download NLTK data if not already present
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    return True

def detect_encoding(uploaded_file):
    """Detect the encoding of the uploaded file"""
    # Read a sample of the file to detect encoding
    uploaded_file.seek(0)  # Reset file pointer
    raw_data = uploaded_file.read(10000)  # Read first 10KB
    uploaded_file.seek(0)  # Reset file pointer again
    
    # Detect encoding
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    confidence = result['confidence']
    
    return encoding, confidence

def read_csv_with_encoding_fallback(uploaded_file):
    """Try to read CSV with different encoding strategies"""
    
    # Strategy 1: Detect encoding automatically
    try:
        encoding, confidence = detect_encoding(uploaded_file)
        if confidence > 0.7:  # High confidence in detection
            df = pd.read_csv(uploaded_file, encoding=encoding)
            return df, f"Successfully read with detected encoding: {encoding} (confidence: {confidence:.2f})"
    except Exception as e:
        st.warning(f"Auto-detection failed: {str(e)}")
    
    # Strategy 2: Try common encodings
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']
    
    for encoding in encodings_to_try:
        try:
            uploaded_file.seek(0)  # Reset file pointer
            df = pd.read_csv(uploaded_file, encoding=encoding)
            return df, f"Successfully read with encoding: {encoding}"
        except UnicodeDecodeError:
            continue
        except Exception as e:
            continue
    
    # Strategy 3: Read with error handling
    try:
        uploaded_file.seek(0)  # Reset file pointer
        df = pd.read_csv(uploaded_file, encoding='utf-8', encoding_errors='replace')
        return df, "Read with UTF-8 and replaced invalid characters with �"
    except Exception as e:
        pass
    
    # Strategy 4: Last resort - binary mode with manual handling
    try:
        uploaded_file.seek(0)
        content = uploaded_file.read()
        # Try to decode as utf-8 with error handling
        text_content = content.decode('utf-8', errors='ignore')
        df = pd.read_csv(io.StringIO(text_content))
        return df, "Read in binary mode with ignored invalid characters"
    except Exception as e:
        raise Exception(f"All encoding strategies failed. Last error: {str(e)}")

def preprocess_text(text):
    """Clean and prepare text for sentence tokenization"""
    if pd.isna(text):
        return ""
    
    # Convert to string and strip whitespace
    text = str(text).strip()
    
    # Replace multiple whitespaces/newlines with single space
    text = re.sub(r'\s+', ' ', text)
    
    return text

def tokenize_sentences_with_hashtags(text, include_hashtags_separate=True):
    """Split text into individual sentences with hashtag handling"""
    if not text:
        return []
    
    if include_hashtags_separate:
        # Extract hashtags and process them separately
        hashtag_pattern = r'(#\w+(?:\s+#\w+)*)'
        
        # Split text by hashtags while preserving them
        parts = re.split(hashtag_pattern, text)
        
        all_sentences = []
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            # Check if this part is hashtags
            if re.match(r'^#\w+(?:\s+#\w+)*$', part):
                # Treat hashtags as separate sentences
                all_sentences.append(part)
            else:
                # Use NLTK sentence tokenizer for regular text
                sentences = nltk.sent_tokenize(part)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence:
                        all_sentences.append(sentence)
        
        return all_sentences
    else:
        # Standard sentence tokenization without special hashtag handling
        sentences = nltk.sent_tokenize(text)
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                clean_sentences.append(sentence)
        return clean_sentences

def transform_data(df, id_column, context_column, include_hashtags_separate=True):
    """Transform data to sentence-tokenized format"""
    
    # Initialize list to store transformed rows
    transformed_rows = []
    
    # Process each record
    for _, row in df.iterrows():
        record_id = row[id_column]
        context_text = preprocess_text(row[context_column])
        
        # Tokenize sentences
        sentences = tokenize_sentences_with_hashtags(context_text, include_hashtags_separate)
        
        # Create rows for each sentence
        for i, sentence in enumerate(sentences, 1):
            transformed_rows.append({
                'ID': record_id,
                'Sentence ID': i,
                'Context': context_text,
                'Statement': sentence
            })
    
    # Create transformed dataframe
    df_transformed = pd.DataFrame(transformed_rows)
    
    return df_transformed

def main():
    st.set_page_config(
        page_title="Text Sentence Tokenizer",
        page_icon="🔄",
        layout="wide"
    )
    
    st.title("🔄 Text Sentence Tokenizer")
    st.markdown("Transform your text data into sentence-level format for analysis")
    
    # Download NLTK data
    with st.spinner("Initializing NLTK resources..."):
        download_nltk_data()
    
    # How to Use section
    with st.expander("📖 How to Use", expanded=True):
        st.markdown("""
        1. **Upload your CSV file** using the file uploader below
        2. **Select ID Column** - Choose the column that uniquely identifies each record
        3. **Select Context Column** - Choose the column containing the text to be transformed
        4. **Configure options** - Choose whether to include hashtags as separate sentences
        5. **Click Transform** - Process your data into sentence-level format
        6. **Download results** - Get your transformed data as a CSV file
        """)
    
    # Output Format section
    with st.expander("📊 Output Format"):
        st.markdown("""
        The transformed data will have the following columns:
        * **ID**: The identifier from your selected ID column
        * **Sentence ID**: Sequential number for each sentence within a record
        * **Context**: The original text from your Context column
        * **Statement**: Individual sentences extracted from the context
        """)
    
    # File upload section
    st.header("📁 Upload Your Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload your CSV file with text data to be tokenized"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file with encoding detection and fallback
            with st.spinner("Reading file and detecting encoding..."):
                df_raw, encoding_message = read_csv_with_encoding_fallback(uploaded_file)
            
            # Show encoding information
            st.success(f"✅ {encoding_message}")
            
            # Display raw data preview
            st.header("📋 Data Preview")
            st.dataframe(df_raw.head(), use_container_width=True)
            st.info(f"Total records: {len(df_raw)} | Columns: {', '.join(df_raw.columns)}")
            
            # Column selection
            st.header("⚙️ Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                id_column = st.selectbox(
                    "Select ID Column",
                    options=df_raw.columns,
                    help="Choose the column that uniquely identifies each record"
                )
            
            with col2:
                context_column = st.selectbox(
                    "Select Context Column", 
                    options=df_raw.columns,
                    help="Choose the column containing the text to be transformed"
                )
            
            # Hashtag option
            include_hashtags = st.checkbox(
                "Include hashtags as separate sentences",
                value=True,
                help="When enabled, hashtags will be treated as individual sentences"
            )
            
            # Validation
            if id_column == context_column:
                st.warning("⚠️ ID and Context columns should be different")
            
            # Preview selection
            if id_column and context_column:
                st.subheader("🔍 Selected Data Preview")
                preview_df = df_raw[[id_column, context_column]].head(3)
                st.dataframe(preview_df, use_container_width=True)
            
            # Transform button
            if st.button("🚀 Transform Data", type="primary"):
                if id_column == context_column:
                    st.error("❌ Please select different columns for ID and Context")
                    st.stop()
                
                with st.spinner("Processing sentences..."):
                    # Transform the data
                    df_transformed = transform_data(
                        df_raw, 
                        id_column, 
                        context_column, 
                        include_hashtags
                    )
                
                # Display results
                st.header("✅ Transformed Data")
                st.dataframe(df_transformed, use_container_width=True)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Records", len(df_raw))
                with col2:
                    st.metric("Total Sentences", len(df_transformed))
                with col3:
                    avg_sentences = len(df_transformed) / len(df_raw) if len(df_raw) > 0 else 0
                    st.metric("Avg Sentences/Record", f"{avg_sentences:.1f}")
                
                # Sample transformation display
                st.header("🔍 Sample Transformation")
                if not df_transformed.empty:
                    sample_id = df_transformed['ID'].iloc[0]
                    sample_data = df_transformed[df_transformed['ID'] == sample_id]
                    
                    st.subheader(f"Record ID: {sample_id}")
                    st.write(f"**Original Text:** {sample_data['Context'].iloc[0]}")
                    st.write("**Tokenized Sentences:**")
                    for _, row in sample_data.iterrows():
                        st.write(f"{row['Sentence ID']}. {row['Statement']}")
                
                # Download section
                st.header("💾 Download Results")
                
                # Convert dataframe to CSV
                csv_buffer = io.StringIO()
                df_transformed.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                filename = f"tokenized_sentences_{id_column}_{context_column}.csv"
                
                st.download_button(
                    label="📥 Download Transformed Data",
                    data=csv_data,
                    file_name=filename,
                    mime="text/csv",
                    help="Download the sentence-tokenized data"
                )
                
                # Display transformation summary
                st.success(f"✅ Successfully processed {len(df_raw)} records into {len(df_transformed)} sentences!")
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("💡 **Troubleshooting tips:**")
            st.markdown("""
            - Make sure your file is a valid CSV format
            - Try saving your file as UTF-8 encoded CSV from Excel or your editor
            - Check if the file contains special characters or was created on a different system
            - If the file is very large, try with a smaller sample first
            """)
    
    else:
        # Show example when no file is uploaded
        st.header("📄 Example")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Data")
            example_input = pd.DataFrame({
                'post_id': ['A123', 'B456'],
                'content': [
                    'Love this new style! Perfect for summer.',
                    'Check out our collection. Amazing quality! #fashion #style'
                ],
                'author': ['user1', 'user2']
            })
            st.dataframe(example_input, use_container_width=True)
            st.caption("Select 'post_id' as ID Column and 'content' as Context Column")
        
        with col2:
            st.subheader("Output Data")
            example_output = pd.DataFrame({
                'ID': ['A123', 'A123', 'B456', 'B456', 'B456'],
                'Sentence ID': [1, 2, 1, 2, 3],
                'Context': [
                    'Love this new style! Perfect for summer.',
                    'Love this new style! Perfect for summer.',
                    'Check out our collection. Amazing quality! #fashion #style',
                    'Check out our collection. Amazing quality! #fashion #style',
                    'Check out our collection. Amazing quality! #fashion #style'
                ],
                'Statement': [
                    'Love this new style!',
                    'Perfect for summer.',
                    'Check out our collection.',
                    'Amazing quality!',
                    '#fashion #style'
                ]
            })
            st.dataframe(example_output, use_container_width=True)
            st.caption("Result with hashtags as separate sentences enabled")

if __name__ == "__main__":
    main()
