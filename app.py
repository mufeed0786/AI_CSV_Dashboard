# ===========================================
# 📊 AI-Powered CSV Data Dashboard
# Full Implementation (Day 1 → Day 7)
# Developer: Mohd Mufeed
# ===========================================

import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from groq import Groq

# -------------------------------
# Load Environment Variables
# -------------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("🚨 GROQ_API_KEY not found in .env file. Please set it before running.")
    st.stop()

# Debugging helper
print("DEBUG: GROQ_API_KEY =", groq_api_key)

# Initialize Groq client
client = Groq(api_key=groq_api_key)

# -------------------------------
# Streamlit Page Configuration
# -------------------------------
st.set_page_config(page_title="AI CSV Dashboard", layout="wide")
st.title("📊 AI-Powered CSV Data Dashboard")
st.write("Upload a CSV file to explore, filter, visualize, and query your data with AI assistance.")

# -------------------------------
# Step 1: File Upload
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload your CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Fix potential serialization issues with Arrow
        for col in df.columns:
            if df[col].dtype == "object":
                try:
                    df[col] = pd.to_datetime(df[col], errors="ignore")
                except Exception:
                    df[col] = df[col].astype(str)

        # -------------------------------
        # Step 2: Data Preview
        # -------------------------------
        st.subheader("🔍 Dataset Preview")
        st.dataframe(df.head())

        # -------------------------------
        # Step 3: Data Summary
        # -------------------------------
        st.subheader("📌 Dataset Summary")
        st.write(f"**Shape:** {df.shape}")
        st.write(f"**Columns:** {list(df.columns)}")
        st.write("**Statistics:**")
        st.write(df.describe(include="all"))

        # -------------------------------
        # Step 4: Download Option
        # -------------------------------
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Processed CSV",
            data=csv,
            file_name="processed_data.csv",
            mime="text/csv"
        )

        # -------------------------------
        # Step 5: Search & Filter
        # -------------------------------
        st.subheader("🔎 Search & Filter Data")
        search_col = st.selectbox("Select column to search", df.columns)
        keyword = st.text_input("Enter keyword")

        if keyword:
            filtered_df = df[df[search_col].astype(str).str.contains(keyword, case=False, na=False)]
            st.write(f"Showing results for **{keyword}**:")
            st.dataframe(filtered_df)
        else:
            filtered_df = df

        # -------------------------------
        # Step 6: Data Visualization
        # -------------------------------
        st.subheader("📊 Data Visualization")
        chart_type = st.selectbox("Choose Chart Type", ["Bar Chart", "Pie Chart", "Heatmap"])

        col_x = st.selectbox("Select column for X-axis", df.columns)
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        col_y = st.selectbox("Select column for Y-axis (numeric)", numeric_cols)

        if chart_type == "Bar Chart":
            fig, ax = plt.subplots()
            sns.barplot(x=df[col_x], y=df[col_y], ax=ax)
            st.pyplot(fig)

        elif chart_type == "Pie Chart":
            pie_data = df[col_x].value_counts()
            fig, ax = plt.subplots()
            ax.pie(pie_data, labels=pie_data.index, autopct="%1.1f%%")
            st.pyplot(fig)

        elif chart_type == "Heatmap":
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        # -------------------------------
        # Step 7: AI Assistance (Insights + Q&A)
        # -------------------------------
        st.subheader("🤖 AI Assistance")

        st.write("### ✨ Automated Insights")
        if st.button("Generate AI Insights"):
            try:
                sample_data = df.head(50).to_csv(index=False)
                prompt = f"""
You are a professional data analyst. Analyze the following dataset sample and give key insights:

Dataset Sample:
{sample_data}

Provide observations about trends, anomalies, missing data, or distributions.
"""
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.1-8b-instant"
                )
                st.success("✅ AI Insights Generated")
                st.write(response.choices[0].message.content)

            except Exception as e:
                st.error(f"Error generating insights: {e}")

        st.write("### 💬 Ask Questions About Your Data")
        user_question = st.text_input("Type your question here:")
        if st.button("Ask AI"):
            if user_question.strip():
                try:
                    sample_data = df.head(50).to_csv(index=False)
                    prompt = f"""
You are a data analyst. The user has provided a dataset sample in CSV format.

Dataset Sample:
{sample_data}

User question: {user_question}

Answer clearly and provide insights. If calculation is needed, estimate based on the sample provided.
"""
                    response = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="llama-3.1-8b-instant"
                    )
                    st.success("✅ Answer from AI")
                    st.write(response.choices[0].message.content)

                except Exception as e:
                    st.error(f"Error generating answer: {e}")
            else:
                st.warning("Please enter a question before clicking Ask AI.")

    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")

else:
    st.info("👆 Please upload a CSV file to get started.")
