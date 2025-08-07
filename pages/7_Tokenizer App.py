import streamlit as st
import pandas as pd
import nltk
import re
import io

# Download NLTK data if not already present
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
        except:
            pass
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt_tab', quiet=True)
        except:
            pass
    return True

def preprocess_text(text, remove_emojis=True, remove_urls=True, normalize_whitespace=True):
    """Clean and prepare text for sentence tokenization with configurable options"""
    if pd.isna(text):
        return ""
    
    # Convert to string and strip whitespace
    text = str(text).strip()
    
    # Remove URLs if requested
    if remove_urls:
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove emojis if requested (basic emoji removal)
    if remove_emojis:
        # Remove common emoji patterns
        text = re.sub(r'[\U0001F600-\U0001F64F]', '', text)  # emoticons
        text = re.sub(r'[\U0001F300-\U0001F5FF]', '', text)  # symbols & pictographs
        text = re.sub(r'[\U0001F680-\U0001F6FF]', '', text)  # transport & map symbols
        text = re.sub(r'[\U0001F1E0-\U0001F1FF]', '', text)  # flags
        text = re.sub(r'[\U00002600-\U000026FF]', '', text)  # miscellaneous symbols
        text = re.sub(r'[\U00002700-\U000027BF]', '', text)  # dingbats
    
    # Normalize whitespace if requested
    if normalize_whitespace:
        text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def tokenize_sentences_with_hashtags(text, include_hashtags_separate=True, min_sentence_length=3):
    """Split text into individual sentences with hashtag handling"""
    if not text or len(text.strip()) < min_sentence_length:
        return []
    
    if include_hashtags_separate:
        # Extract hashtags and process them separately
        hashtag_pattern = r'(#\w+(?:\s+#\w+)*)'
        
        # Split text by hashtags while preserving them
        parts = re.split(hashtag_pattern, text)
        
        all_sentences = []
        
        for part in parts:
            part = part.strip()
            if not part or len(part) < min_sentence_length:
                continue
                
            # Check if this part is hashtags
            if re.match(r'^#\w+(?:\s+#\w+)*$', part):
                # Treat hashtags as separate sentences
                all_sentences.append(part)
            else:
                # Use NLTK sentence tokenizer for regular text
                try:
                    sentences = nltk.sent_tokenize(part)
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if sentence and len(sentence) >= min_sentence_length:
                            all_sentences.append(sentence)
                except:
                    # Fallback to simple period splitting if NLTK fails
                    sentences = part.split('.')
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if sentence and len(sentence) >= min_sentence_length:
                            all_sentences.append(sentence)
        
        return all_sentences
    else:
        # Standard sentence tokenization without special hashtag handling
        try:
            sentences = nltk.sent_tokenize(text)
        except:
            # Fallback to simple period splitting
            sentences = text.split('.')
        
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) >= min_sentence_length:
                clean_sentences.append(sentence)
        return clean_sentences

def transform_data(df, id_column, context_column, preprocessing_options, tokenization_options):
    """Transform data to sentence-tokenized format"""
    
    # Initialize list to store transformed rows
    transformed_rows = []
    
    # Track processing statistics
    processed_count = 0
    error_count = 0
    
    # Process each record
    for idx, row in df.iterrows():
        try:
            record_id = row[id_column]
            
            # Preprocess text with options
            context_text = preprocess_text(
                row[context_column],
                remove_emojis=preprocessing_options.get('remove_emojis', True),
                remove_urls=preprocessing_options.get('remove_urls', True),
                normalize_whitespace=preprocessing_options.get('normalize_whitespace', True)
            )
            
            # Skip empty records
            if not context_text:
                continue
            
            # Tokenize sentences with options
            sentences = tokenize_sentences_with_hashtags(
                context_text, 
                include_hashtags_separate=tokenization_options.get('include_hashtags_separate', True),
                min_sentence_length=tokenization_options.get('min_sentence_length', 3)
            )
            
            # Create rows for each sentence
            for i, sentence in enumerate(sentences, 1):
                transformed_rows.append({
                    'ID': record_id,
                    'Sentence ID': i,
                    'Context': row[context_column],  # Keep original context
                    'Statement': sentence
                })
            
            processed_count += 1
            
        except Exception as e:
            error_count += 1
            st.warning(f"Error processing row {idx + 1}: {str(e)}")
            continue
    
    # Create transformed dataframe
    df_transformed = pd.DataFrame(transformed_rows)
    
    return df_transformed, processed_count, error_count

def main():
    st.set_page_config(
        page_title="Instagram Posts Dataset Processor",
        page_icon="📱",
        layout="wide"
    )
    
    st.title("📱 Instagram Posts Dataset Processor")
    st.markdown("Transform your Instagram posts data into sentence-level format for analysis")
    
    # Download NLTK data
    with st.spinner("Initializing NLTK resources..."):
        download_nltk_data()
    
    # How to Use section
    with st.expander("📖 How to Use", expanded=True):
        st.markdown("""
        1. **Choose your Instagram posts dataset** - Upload your CSV file
        2. **Adjust preprocessing options** - Configure text cleaning settings
        3. **Map columns** - Select caption and ID columns
        4. **Click to transform your data** - Process into sentence format
        5. **Save your processed results** - Download the transformed data
        """)
    
    # Output Format section
    with st.expander("📊 Output Format"):
        st.markdown("""
        The processed data will have the following structure:
        * **ID**: Post identifier
        * **Sentence ID**: Sentence number within post  
        * **Context**: Original caption text
        * **Statement**: Cleaned individual sentence
        """)
    
    # File upload section
    st.header("📁 Choose Your Instagram Posts Dataset")
    uploaded_file = st.file_uploader(
        "Upload CSV file with Instagram posts data",
        type="csv",
        help="Your CSV should contain columns for post IDs and captions/text content"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file with error handling
            try:
                df_raw = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                df_raw = pd.read_csv(uploaded_file, encoding='latin-1')
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
                st.stop()
            
            # Display raw data preview
            st.header("📋 Dataset Preview")
            st.dataframe(df_raw.head(), use_container_width=True)
            st.info(f"📊 Total records: {len(df_raw)} | Columns: {', '.join(df_raw.columns)}")
            
            # Preprocessing Options
            st.header("⚙️ Adjust Preprocessing Options")
            
            preprocessing_col1, preprocessing_col2, preprocessing_col3 = st.columns(3)
            
            with preprocessing_col1:
                remove_emojis = st.checkbox("Remove emojis", value=True, help="Remove emoji characters from text")
            
            with preprocessing_col2:
                remove_urls = st.checkbox("Remove URLs", value=True, help="Remove web links from captions")
            
            with preprocessing_col3:
                normalize_whitespace = st.checkbox("Normalize whitespace", value=True, help="Replace multiple spaces/newlines with single spaces")
            
            # Advanced tokenization options
            with st.expander("🔧 Advanced Options"):
                col_adv1, col_adv2 = st.columns(2)
                with col_adv1:
                    include_hashtags = st.checkbox(
                        "Include hashtags as separate sentences",
                        value=True,
                        help="When enabled, hashtags will be treated as individual sentences"
                    )
                with col_adv2:
                    min_sentence_length = st.slider(
                        "Minimum sentence length",
                        min_value=1,
                        max_value=10,
                        value=3,
                        help="Minimum number of characters for a valid sentence"
                    )
            
            # Column mapping section
            st.header("🗂️ Map Columns")
            
            col1, col2 = st.columns(2)
            
            with col1:
                id_column = st.selectbox(
                    "Select ID Column (Post identifier)",
                    options=df_raw.columns,
                    help="Choose the column that contains unique post identifiers"
                )
            
            with col2:
                context_column = st.selectbox(
                    "Select Caption Column (Text content)", 
                    options=df_raw.columns,
                    help="Choose the column containing the post captions/text to be processed"
                )
            
            # Validation
            if id_column == context_column:
                st.warning("⚠️ ID and Caption columns should be different")
            
            # Preview selection
            if id_column and context_column and id_column != context_column:
                st.subheader("🔍 Selected Data Preview")
                preview_df = df_raw[[id_column, context_column]].head(3)
                st.dataframe(preview_df, use_container_width=True)
                
                # Show sample preprocessing result
                sample_text = str(df_raw[context_column].iloc[0]) if len(df_raw) > 0 else ""
                if sample_text:
                    processed_sample = preprocess_text(sample_text, remove_emojis, remove_urls, normalize_whitespace)
                    st.write("**Sample preprocessing result:**")
                    st.write(f"Original: `{sample_text[:100]}{'...' if len(sample_text) > 100 else ''}`")
                    st.write(f"Processed: `{processed_sample[:100]}{'...' if len(processed_sample) > 100 else ''}`")
            
            # Transform button
            st.header("🚀 Click to Transform Your Data")
            if st.button("Transform Data", type="primary", use_container_width=True):
                if id_column == context_column:
                    st.error("❌ Please select different columns for ID and Caption")
                    st.stop()
                
                with st.spinner("Processing sentences..."):
                    # Prepare options dictionaries
                    preprocessing_options = {
                        'remove_emojis': remove_emojis,
                        'remove_urls': remove_urls,
                        'normalize_whitespace': normalize_whitespace
                    }
                    
                    tokenization_options = {
                        'include_hashtags_separate': include_hashtags,
                        'min_sentence_length': min_sentence_length
                    }
                    
                    # Transform the data
                    df_transformed, processed_count, error_count = transform_data(
                        df_raw, 
                        id_column, 
                        context_column, 
                        preprocessing_options,
                        tokenization_options
                    )
                
                if df_transformed.empty:
                    st.error("❌ No data was processed. Please check your data and settings.")
                    st.stop()
                
                # Display results
                st.header("✅ Processed Results")
                st.dataframe(df_transformed, use_container_width=True)
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Original Records", len(df_raw))
                with col2:
                    st.metric("Processed Records", processed_count)
                with col3:
                    st.metric("Total Sentences", len(df_transformed))
                with col4:
                    avg_sentences = len(df_transformed) / processed_count if processed_count > 0 else 0
                    st.metric("Avg Sentences/Post", f"{avg_sentences:.1f}")
                
                if error_count > 0:
                    st.warning(f"⚠️ {error_count} records had processing errors")
                
                # Sample transformation display
                st.header("🔍 Sample Transformation")
                if not df_transformed.empty:
                    sample_id = df_transformed['ID'].iloc[0]
                    sample_data = df_transformed[df_transformed['ID'] == sample_id]
                    
                    st.subheader(f"Post ID: {sample_id}")
                    st.write(f"**Original Caption:** {sample_data['Context'].iloc[0]}")
                    st.write("**Extracted Sentences:**")
                    for _, row in sample_data.iterrows():
                        st.write(f"{row['Sentence ID']}. {row['Statement']}")
                
                # Download section
                st.header("💾 Save Your Processed Results")
                
                # Convert dataframe to CSV
                csv_buffer = io.StringIO()
                df_transformed.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                # Generate filename based on original filename and columns
                original_name = uploaded_file.name.split('.')[0] if hasattr(uploaded_file, 'name') else 'processed'
                filename = f"{original_name}_processed_sentences.csv"
                
                col_download1, col_download2 = st.columns([2, 1])
                
                with col_download1:
                    st.download_button(
                        label="📥 Download Processed Results",
                        data=csv_data,
                        file_name=filename,
                        mime="text/csv",
                        help="Download the sentence-tokenized Instagram posts data",
                        use_container_width=True
                    )
                
                with col_download2:
                    st.metric("File Size", f"{len(csv_data.encode('utf-8')) / 1024:.1f} KB")
                
                # Display transformation summary
                st.success(f"✅ Successfully processed {processed_count} Instagram posts into {len(df_transformed)} sentences!")
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your CSV file is properly formatted with appropriate encoding.")
    
    else:
        # Show example when no file is uploaded
        st.header("📄 Example")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Data (Instagram Posts)")
            example_input = pd.DataFrame({
                'post_id': ['IG_001', 'IG_002', 'IG_003'],
                'caption': [
                    'Love this new summer style! Perfect for beach days. 🌞 #fashion #summer',
                    'Check out our latest collection. Amazing quality and design! Visit our website: www.example.com',
                    'Grateful for all the support! Thank you everyone. #grateful #community #love'
                ],
                'author': ['user1', 'user2', 'user3'],
                'likes': [245, 189, 342]
            })
            st.dataframe(example_input, use_container_width=True)
            st.caption("Select 'post_id' as ID Column and 'caption' as Caption Column")
        
        with col2:
            st.subheader("Output Data (Processed)")
            example_output = pd.DataFrame({
                'ID': ['IG_001', 'IG_001', 'IG_001', 'IG_002', 'IG_002', 'IG_002', 'IG_003', 'IG_003', 'IG_003'],
                'Sentence ID': [1, 2, 3, 1, 2, 3, 1, 2, 3],
                'Context': [
                    'Love this new summer style! Perfect for beach days. 🌞 #fashion #summer',
                    'Love this new summer style! Perfect for beach days. 🌞 #fashion #summer',
                    'Love this new summer style! Perfect for beach days. 🌞 #fashion #summer',
                    'Check out our latest collection. Amazing quality and design! Visit our website: www.example.com',
                    'Check out our latest collection. Amazing quality and design! Visit our website: www.example.com',
                    'Check out our latest collection. Amazing quality and design! Visit our website: www.example.com',
                    'Grateful for all the support! Thank you everyone. #grateful #community #love',
                    'Grateful for all the support! Thank you everyone. #grateful #community #love',
                    'Grateful for all the support! Thank you everyone. #grateful #community #love'
                ],
                'Statement': [
                    'Love this new summer style!',
                    'Perfect for beach days.',
                    '#fashion #summer',
                    'Check out our latest collection.',
                    'Amazing quality and design!',
                    'Visit our website: www.example.com',
                    'Grateful for all the support!',
                    'Thank you everyone.',
                    '#grateful #community #love'
                ]
            })
            st.dataframe(example_output, use_container_width=True)
            st.caption("Result with preprocessing and hashtag separation enabled")

if __name__ == "__main__":
    main()
