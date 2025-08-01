import streamlit as st
import pandas as pd
import nltk
import re
import io

# Download NLTK data if not already present
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    return True

def preprocess_text(text):
    """Clean and prepare text for sentence tokenization"""
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def tokenize_sentences(text):
    """Split text into individual sentences"""
    if not text:
        return []
    sentences = nltk.sent_tokenize(text)
    clean_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            clean_sentences.append(sentence)
    return clean_sentences

def create_rolling_context(statements, window_size):
    """Create rolling window context for each statement"""
    contexts = []
    for i in range(len(statements)):
        start_idx = max(0, i - window_size + 1)
        context_window = statements[start_idx:i+1]
        contexts.append(' '.join(context_window))
    return contexts

def process_data(df, statement_level, context_type, window_size=3):
    """Process data based on user selections"""
    
    processed_rows = []
    
    # Group by ID (post/chat level)
    for post_id, group in df.groupby('ID'):
        
        if statement_level == "Sentence Level":
            # Tokenize all turns into sentences
            all_sentences = []
            sentence_to_turn = []
            sentence_to_speaker = []
            
            for _, row in group.iterrows():
                turn_text = preprocess_text(row['Statement'])
                sentences = tokenize_sentences(turn_text)
                
                for sentence in sentences:
                    all_sentences.append(sentence)
                    sentence_to_turn.append(row['Turn'])
                    sentence_to_speaker.append(row.get('Speaker', 'Unknown'))
            
            # Create contexts based on selection
            if context_type == "Rolling Window":
                contexts = create_rolling_context(all_sentences, window_size)
            else:  # Whole Post
                full_context = ' '.join(all_sentences)
                contexts = [full_context] * len(all_sentences)
            
            # Create rows for each sentence
            for i, (sentence, context) in enumerate(zip(all_sentences, contexts)):
                processed_rows.append({
                    'ID': post_id,
                    'Turn': sentence_to_turn[i],
                    'Sentence ID': i + 1,
                    'Context': context,
                    'Statement': sentence,
                    'Speaker': sentence_to_speaker[i],
                    'Statement Level': 'Sentence',
                    'Context Type': context_type
                })
        
        elif statement_level == "Turn Level":
            # Use turns as statements
            all_turns = []
            turn_speakers = []
            
            for _, row in group.iterrows():
                turn_text = preprocess_text(row['Statement'])
                all_turns.append(turn_text)
                turn_speakers.append(row.get('Speaker', 'Unknown'))
            
            # Create contexts based on selection
            if context_type == "Rolling Window":
                contexts = create_rolling_context(all_turns, window_size)
            else:  # Whole Post
                full_context = ' '.join(all_turns)
                contexts = [full_context] * len(all_turns)
            
            # Create rows for each turn
            for i, (turn, context) in enumerate(zip(all_turns, contexts)):
                processed_rows.append({
                    'ID': post_id,
                    'Turn': group.iloc[i]['Turn'],
                    'Sentence ID': i + 1,
                    'Context': context,
                    'Statement': turn,
                    'Speaker': turn_speakers[i],
                    'Statement Level': 'Turn',
                    'Context Type': context_type
                })
        
        else:  # Post Level
            # Use entire post as single statement
            all_text = []
            speakers = []
            
            for _, row in group.iterrows():
                all_text.append(preprocess_text(row['Statement']))
                speakers.append(row.get('Speaker', 'Unknown'))
            
            full_statement = ' '.join(all_text)
            
            processed_rows.append({
                'ID': post_id,
                'Turn': 'All',
                'Sentence ID': 1,
                'Context': full_statement,  # For post level, context = statement
                'Statement': full_statement,
                'Speaker': ', '.join(set(speakers)),
                'Statement Level': 'Post',
                'Context Type': context_type
            })
    
    return pd.DataFrame(processed_rows)

def main():
    st.set_page_config(
        page_title="Text Preprocessing App",
        page_icon="🔄",
        layout="wide"
    )
    
    st.title("🔄 Text Preprocessing App with Rolling Context")
    st.markdown("Configure context and statement types for text classification experiments")
    
    # Download NLTK data
    with st.spinner("Initializing NLTK resources..."):
        download_nltk_data()
    
    # Configuration sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Statement level selection
        statement_level = st.selectbox(
            "📝 Statement Level",
            ["Sentence Level", "Turn Level", "Post Level"],
            help="Choose how to split text into statements"
        )
        
        # Context type selection
        context_type = st.selectbox(
            "📋 Context Type",
            ["Rolling Window", "Whole Post"],
            help="Choose context scope for each statement"
        )
        
        # Window size (only for rolling window)
        if context_type == "Rolling Window":
            window_size = st.slider(
                "🪟 Window Size",
                min_value=1,
                max_value=10,
                value=3,
                help="Number of previous statements to include in context"
            )
        else:
            window_size = None
        
        # Speaker filter (optional)
        speaker_filter = st.multiselect(
            "👥 Filter by Speaker",
            ["customer", "salesperson"],
            help="Optional: filter results by speaker type"
        )
        
        st.header("📊 Expected Input")
        st.markdown("""
        **Required columns:**
        - `ID`: Post/Chat identifier
        - `Turn`: Turn number
        - `Statement`: Text content
        - `Speaker`: customer/salesperson (optional)
        """)
    
    # File upload section
    st.header("📁 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload your dataset with ID, Turn, Statement columns"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df_raw = pd.read_csv(uploaded_file)
            
            # Validate required columns
            required_columns = ['ID', 'Turn', 'Statement']
            missing_columns = [col for col in required_columns if col not in df_raw.columns]
            
            if missing_columns:
                st.error(f"❌ Missing required columns: {missing_columns}")
                st.stop()
            
            # Add Speaker column if not present
            if 'Speaker' not in df_raw.columns:
                st.warning("⚠️ No 'Speaker' column found. Adding default values.")
                df_raw['Speaker'] = 'Unknown'
            
            # Display raw data preview
            st.header("📋 Raw Data Preview")
            st.dataframe(df_raw.head(10), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df_raw))
            with col2:
                st.metric("Unique IDs", df_raw['ID'].nunique())
            with col3:
                st.metric("Total Turns", df_raw['Turn'].nunique())
            
            # Configuration summary
            st.header("🔧 Processing Configuration")
            config_col1, config_col2 = st.columns(2)
            
            with config_col1:
                st.info(f"**Statement Level:** {statement_level}")
                st.info(f"**Context Type:** {context_type}")
                if window_size:
                    st.info(f"**Window Size:** {window_size}")
            
            with config_col2:
                if speaker_filter:
                    st.info(f"**Speaker Filter:** {', '.join(speaker_filter)}")
                else:
                    st.info("**Speaker Filter:** None")
            
            # Process button
            if st.button("🚀 Process Data", type="primary"):
                with st.spinner("Processing data with selected configuration..."):
                    # Apply speaker filter if selected
                    if speaker_filter:
                        df_filtered = df_raw[df_raw['Speaker'].isin(speaker_filter)]
                        if df_filtered.empty:
                            st.error("No data matches the selected speaker filter!")
                            st.stop()
                    else:
                        df_filtered = df_raw
                    
                    # Process the data
                    df_processed = process_data(
                        df_filtered, 
                        statement_level, 
                        context_type, 
                        window_size or 3
                    )
                
                # Display results
                st.header("✅ Processed Data")
                st.dataframe(df_processed, use_container_width=True)
                
                # Results summary
                st.header("📊 Processing Results")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Input Records", len(df_filtered))
                with col2:
                    st.metric("Output Records", len(df_processed))
                with col3:
                    st.metric("Processed IDs", df_processed['ID'].nunique())
                with col4:
                    expansion_ratio = len(df_processed) / len(df_filtered) if len(df_filtered) > 0 else 0
                    st.metric("Expansion Ratio", f"{expansion_ratio:.2f}x")
                
                # Sample transformation
                if not df_processed.empty:
                    st.header("🔍 Sample Transformation")
                    sample_id = df_processed['ID'].iloc[0]
                    sample_data = df_processed[df_processed['ID'] == sample_id].head(3)
                    
                    for _, row in sample_data.iterrows():
                        with st.expander(f"ID: {row['ID']}, Turn: {row['Turn']}, Sentence: {row['Sentence ID']}"):
                            st.write(f"**Statement:** {row['Statement']}")
                            st.write(f"**Context:** {row['Context']}")
                            st.write(f"**Speaker:** {row['Speaker']}")
                
                # Download section
                st.header("💾 Download Results")
                
                # Convert dataframe to CSV
                csv_buffer = io.StringIO()
                df_processed.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                filename = f"processed_data_{statement_level.lower().replace(' ', '_')}_{context_type.lower().replace(' ', '_')}.csv"
                
                st.download_button(
                    label="📥 Download Processed Data",
                    data=csv_data,
                    file_name=filename,
                    mime="text/csv",
                    help="Download the processed dataset"
                )
                
                st.success(f"✅ Successfully processed {len(df_filtered)} records into {len(df_processed)} statements!")
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please check that your CSV file has the correct format and required columns.")
    
    else:
        # Example data format
        st.header("📄 Example Data Format")
        
        example_data = pd.DataFrame({
            'ID': ['post_001', 'post_001', 'post_001', 'post_002', 'post_002'],
            'Turn': [1, 2, 3, 1, 2],
            'Statement': [
                'Hi there! I love your new collection.',
                'Thank you! Which piece caught your eye?',
                'The blue dress is perfect for summer events.',
                'Do you have this in size medium?',
                'Yes, we have it in stock. Would you like me to reserve it?'
            ],
            'Speaker': ['customer', 'salesperson', 'customer', 'customer', 'salesperson']
        })
        
        st.dataframe(example_data, use_container_width=True)
        
        st.header("🎯 Use Cases")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sentence + Rolling Window")
            st.markdown("""
            - **Best for:** Fine-grained analysis
            - **Context:** Previous 2-3 sentences
            - **Statement:** Individual sentences
            - **Use case:** Intent classification, sentiment analysis
            """)
        
        with col2:
            st.subheader("Turn + Whole Post")
            st.markdown("""
            - **Best for:** Conversation analysis
            - **Context:** Entire conversation
            - **Statement:** Complete turns
            - **Use case:** Topic modeling, conversation flow
            """)

if __name__ == "__main__":
    main()
