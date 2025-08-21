import streamlit as st
import os
import google.generativeai as genai
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import io
import plotly.express as px
import asyncio
import os
from dotenv import load_dotenv


api_key = st.secrets["API_KEY"]


#loading and accessing a key locally
#load_dotenv()


# Set Streamlit page configuration for a wider layout and custom styling
st.set_page_config(
    layout="wide",
    page_title="Sentiment Analysis Dashboard",
    page_icon="📈"
)

# --- Sentiment Analysis Functions ---
analyzer = SentimentIntensityAnalyzer()

def classify_sentiment_vader(text):
    """
    Classifies a single text string's sentiment using VADER.
    Returns the sentiment, confidence, and raw scores.
    """
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        sentiment = "POSITIVE"
    elif compound <= -0.05:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"
    confidence = abs(compound)
    return sentiment, confidence, scores

async def get_gemini_explanation(text, sentiment, scores):
    """
    Generates a natural language explanation for sentiment using Google Gemini.
    """

    #api_key = os.getenv("API_KEY")
    
    try:
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')

        # Craft the prompt for the Generative AI model
        prompt = f"""
        Analyze the following text and its sentiment scores. Provide a clear, concise explanation in a conversational tone and keep it short.

        Text: "{text}"
        Overall Sentiment: {sentiment}
        Scores: Positive={scores['pos']:.2f}, Negative={scores['neg']:.2f}, Neutral={scores['neu']:.2f}

        Explain the reasoning behind this sentiment. Mention which words or lack of emotional language likely contributed to the scores and keep explanations short.
        """

        # Generate the explanation from Gemini
        response = await asyncio.to_thread(model.generate_content, prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while calling the Gemini API: {e}"

async def batch_classify_vader_with_gemini(text_list):
    """
    Analyzes sentiment for a list of texts using VADER and generates Gemini explanations.
    Returns a pandas DataFrame with all results.
    """
    results = []
    tasks = []

    for text in text_list:
        sentiment, confidence, scores = classify_sentiment_vader(text)
        tasks.append(get_gemini_explanation(text, sentiment, scores))
        results.append({
            'Text': text,
            'Sentiment': sentiment,
            'Confidence': confidence,
            'Negative': scores['neg'],
            'Neutral': scores['neu'],
            'Positive': scores['pos'],
            'Explanation': "" # Placeholder for the explanation
        })

    # Run all Gemini explanation tasks concurrently
    explanations = await asyncio.gather(*tasks)

    # Populate the results DataFrame with the explanations
    for i, explanation in enumerate(explanations):
        results[i]['Explanation'] = explanation
        
    return pd.DataFrame(results)

# --- Dashboard Layout and Logic ---

st.markdown("""
    <style>
    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        background-color: #2962ff;
        color: white;
        font-weight: bold;
    }
    .st-emotion-cache-1r6dm7m {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .st-expander>div>p {
        margin: 0;
        font-size: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Sentiment Analysis Dashboard 📈")
st.markdown("Use this dashboard to analyze sentiment from text. You can either enter text directly or upload a file.")
st.markdown("---")

# Create columns for layout
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Input Method")
    option = st.radio("Choose input method:", ('Direct Text Entry', 'Upload File'))

with col2:
    if option == 'Direct Text Entry':
        st.header("Single Text Analysis")
        text_input = st.text_area("Enter text to analyze:")
        if st.button("Analyze"):
            if text_input.strip():
                sentiment, confidence, scores = classify_sentiment_vader(text_input)
                
                # Use a container for the results to give it a card-like appearance
                with st.container(border=True):
                    st.markdown(f"### Results for Your Text")
                    st.metric(label="Overall Sentiment", value=sentiment)

                with st.spinner("Generating sentiment explanation..."):
                    explanation = asyncio.run(get_gemini_explanation(text_input, sentiment, scores))
                
                # Use an expander for the explanation to make it pop up
                with st.expander("Show Sentiment Explanation"):
                    st.info(explanation, icon="🤖")

                st.markdown("### Sentiment Distribution Dashboard")
                if 'compound' in scores:
                    del scores['compound']
                scores_df = pd.DataFrame([scores]).T.reset_index()
                scores_df.columns = ['Score Type', 'Value']
                
                fig_single_bar = px.bar(
                    scores_df, 
                    x='Score Type', 
                    y='Value', 
                    title='Positive, Neutral, and Negative Scores',
                    color='Score Type',
                    color_discrete_map={
                        'Positive': 'green', 
                        'Negative': 'red', 
                        'Neutral': 'gray'
                    },
                    labels={'Value': 'Score', 'Score Type': 'Sentiment Type'},
                    text_auto=True 
                )
                fig_single_bar.update_traces(textposition="outside")
                fig_single_bar.update_layout(hovermode='x unified')
                st.plotly_chart(fig_single_bar, use_container_width=True)
                
            else:
                st.warning("Please enter some text.")
    else:
        st.header("Batch Analysis from File")
        uploaded_file = st.file_uploader("Upload a TXT or CSV file", type=['txt', 'csv'])
        if uploaded_file is not None:
            if st.button("Generate sentiment explanation"):
                with st.spinner('Analyzing sentiments...'):
                    texts = []
                    
                    if uploaded_file.type == "text/plain":
                        texts = [line.strip() for line in io.StringIO(uploaded_file.getvalue().decode("utf-8")) if line.strip()]
                    elif uploaded_file.type == "text/csv":
                        df = pd.read_csv(uploaded_file)
                        if 'review' in df.columns:
                            texts = df['review'].dropna().astype(str).tolist()
                        else:
                            texts = df.iloc[:, 0].dropna().astype(str).tolist()
                    
                    if texts:
                        results_df = asyncio.run(batch_classify_vader_with_gemini(texts))
                        
                        st.success("Analysis complete!")
                        
                        # Use a container for the batch analysis results
                        with st.container(border=True):
                            st.dataframe(results_df, use_container_width=True)
                            
                            with st.expander("Show Sentiment Explanations"):
                                for index, row in results_df.iterrows():
                                    st.info(f"**Text:** `{row['Text'][:50]}...`\n\n**Sentiment:** {row['Sentiment']}\n\n**Explanation:** {row['Explanation']}", icon="🤖")

                            st.markdown("---")
                            
                            st.markdown("### Sentiment Distribution Dashboard")
                            
                            # Only displaying the Average Sentiment Scores graph
                            avg_scores = results_df[['Positive', 'Negative', 'Neutral']].mean().reset_index()
                            avg_scores.columns = ['Score Type', 'Average Score']
                            
                            fig_avg_bar = px.bar(
                                avg_scores,
                                x='Score Type',
                                y='Average Score',
                                color='Score Type',
                                color_discrete_map={
                                    'Positive': 'green',
                                    'Negative': 'red',
                                    'Neutral': 'gray'
                                },
                                text_auto=True
                            )
                            fig_avg_bar.update_traces(textposition="outside")
                            fig_avg_bar.update_layout(hovermode='x unified')
                            st.plotly_chart(fig_avg_bar, use_container_width=True)

                    else:
                        st.warning("No valid texts found in the file.")
        else:
            st.info("Upload a file to begin batch analysis.")