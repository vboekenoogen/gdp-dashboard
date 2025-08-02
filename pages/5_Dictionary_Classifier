import streamlit as st
import pandas as pd
import io
import json

st.title("🧠 Dictionary Classifier App")
st.markdown("""
Upload a CSV, select the column with text, and input your dictionary as either JSON or lines like:
`Luxury: elegant, timeless, classic`.
Then run the analysis to get category predictions based on keyword matches.
""")

uploaded_file = st.file_uploader("📁 Upload CSV", type="csv")

df_loaded = None
result_df = None

if uploaded_file:
    try:
        df_loaded = pd.read_csv(uploaded_file)
        st.success("CSV uploaded successfully!")
        text_col = st.selectbox("Text Column:", df_loaded.columns)

        dict_input = st.text_area(
            "Dictionary:",
            value='Luxury: elegant, timeless, refined, classic, sophisticated, luxury, polished',
            help='Enter JSON or lines like "Category: keyword1, keyword2, ..."'
        )

        if st.button("🔍 Run Analysis"):
            dict_text = dict_input.strip()
            categories = {}
            try:
                parsed = json.loads(dict_text)
                if isinstance(parsed, dict):
                    for k, v in parsed.items():
                        if isinstance(v, str):
                            v = [v]
                        categories[k] = [str(x).strip() for x in v]
            except:
                for line in dict_text.splitlines():
                    if ':' in line:
                        cat, terms = line.split(':', 1)
                        categories[cat.strip()] = [w.strip() for w in terms.split(',') if w.strip()]

            if not categories:
                st.error("❌ Invalid dictionary format.")
            else:
                results = []
                for txt in df_loaded[text_col].astype(str):
                    txt_l = txt.lower()
                    best_cat, max_count = None, 0
                    for cat, words in categories.items():
                        count = sum(txt_l.count(w.lower()) for w in words if w)
                        if count > max_count:
                            best_cat, max_count = cat, count
                    results.append(best_cat)

                result_df = df_loaded.copy()
                result_df['Predicted_Category'] = results

                st.success("✅ Analysis complete! Sample results:")
                st.dataframe(result_df.head())

                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Download Results",
                    data=csv,
                    file_name='classification_results.csv',
                    mime='text/csv'
                )
    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")
