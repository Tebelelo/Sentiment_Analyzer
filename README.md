**📈 Sentiment Analysis Dashboard**

This is an interactive Streamlit web app that performs sentiment analysis on input text using VADER (Valence Aware Dictionary and sEntiment Reasoner) and generates short, conversational explanations using Google Gemini AI.

**Users can:**

Analyze individual pieces of text

Upload .txt for batch analysis

View sentiment scores, explanations, and visualizations



**🚀 Features**

📊 Sentiment classification using VADER

🤖 Explanation generation with Google Gemini (Generative AI)

📁 File upload support for bulk text analysis

📉 Visual summaries with Plotly

🌐 Ready for Streamlit Cloud deployment



**🧰 Technologies Used**

Streamlit
 – UI

VADER Sentiment
 – sentiment analysis

Google Generative AI
 – explanation generation

Plotly
 – data visualization

Python Dotenv
 – environment variable loading

 

**📦 Requirements**

Create a requirements.txt file with the following content:

streamlit
google-generativeai
vaderSentiment
pandas
plotly
python-dotenv


Or install manually:

pip install streamlit google-generativeai vaderSentiment pandas plotly python-dotenv


**🛠️ Local Setup Instructions**
1. Clone the repo
git clone https://github.com/Tebelelo/Sentiment_Analyzer.git
cd sentiment-dashboard

2. Create .env (for local development only)

  .Create a .env file and add your Google API key:

  .API_KEY=your_gemini_api_key_here

  Note: This file should NOT be committed to version control.

3. Run the app
streamlit run app.py


