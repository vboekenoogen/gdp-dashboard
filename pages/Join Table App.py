# csv_join_app.py
# Streamlit app: Upload two CSVs, choose join type & columns, download result.

import streamlit as st
import pandas as pd
import io

# ---------- Page setup ----------
st.set_page_config(page_title="CSV Join Tool", page_icon="📂")
st.title("📂 CSV Join Tool")
st.markdown(
    "Upload two CSV files, choose your join type and columns, then download the result."
)

# ---------- File upload ----------
uploaded_file1 = st.file_uploader("Step 1️⃣ – Upload **first** CSV file", type="csv", key="csv1")
uploaded_file2 = st.file_uploader("Step 1️⃣ – Upload **second** CSV file", type="csv", key="csv2")

# ---------- Session-state for result ----------
if "joined_df" not in st.session_state:
    st.session_state["joined_df"] = pd.DataFrame()

# ---------- Proceed once both files are uploaded ----------
if uploaded_file1 and uploaded_file2:
    try:
        df1 = pd.read_csv(uploaded_file1)
        df2 = pd.read_csv(uploaded_file2)
    except Exception as e:
        st.error(f"❌ Error reading CSV files: {e}")
        st.stop()

    st.divider()
    st.subheader("Step 2️⃣ – Select Join Options")

    join_type = st.selectbox("🔀 Join type", ("inner", "left", "right", "outer"), index=0)

    col1, col2 = st.columns(2)
    with col1:
        join_column_1 = st.selectbox("📄 Column from CSV 1", df1.columns.tolist())
    with col2:
        join_column_2 = st.selectbox("📄 Column from CSV 2", df2.columns.tolist())

    if st.button("🚀 Join CSVs"):
        try:
            st.session_state["joined_df"] = pd.merge(
                df1,
                df2,
                how=join_type,
                left_on=join_column_1,
                right_on=join_column_2,
            )
            st.success(
                f"✅ Join successful! {len(st.session_state['joined_df'])} rows in the result."
            )
            st.dataframe(
                st.session_state["joined_df"].head(100),
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"❌ Error while joining: {e}")

    # ---------- Download button ----------
    if not st.session_state["joined_df"].empty:
        csv_buffer = io.StringIO()
        st.session_state["joined_df"].to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download Joined CSV",
            data=csv_buffer.getvalue(),
            file_name="joined_result.csv",
            mime="text/csv",
            use_container_width=True,
        )
else:
    st.info("👆 Please upload **both** CSV files to proceed.")
