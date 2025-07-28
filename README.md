#📊 AI-Powered CSV Data Dashboard  

A Streamlit-based interactive dashboard that allows users to upload CSV files, explore data visually, and even ask questions from the dataset using AI (powered by GROQ LLM).  



## 🚀 Features  
- 📂 Upload CSV files easily  
- 🔎 Automatic data preview and summary  
- 📈 Interactive charts and data visualizations  
- 🤖 AI Assistant to ask questions from your dataset  
- ⚡ Real-time analysis with Pandas & Streamlit  
- 🔐 Secure `.env` file for API key management  



## 🛠️ Tech Stack  
- **Python 3.12**  
- **Pandas** – Data processing  
- **Streamlit** – Dashboard framework  
- **Plotly / Matplotlib** – Visualizations  
- **Groq API (LLM)** – AI-powered insights  
- **Git & GitHub** – Version control and deployment  



## 📂 Project Structure  

bash
AI_CSV_Dashboard/
│── app.py           # Main Streamlit app
│── .env             # API keys & config
│── data/
│    └── mydata.csv  # Sample dataset
│── README.md        # Project documentation

⚙️ Installation
Clone the repository
 git clone https://github.com/mufeed0786/AI_CSV_Dashboard.git
 cd AI_CSV_Dashboard

Create a virtual environment
 python -m venv venv
 venv\Scripts\activate    # On Windows


Install dependencies
 pip install -r requirements.txt

Add your GROQ API key in .env file
 GROQ_API_KEY=your_api_key_here

Run the app
 streamlit run app.py

📊 Usage
Upload your CSV file.
View summary statistics & data preview.
Generate visualizations (charts, plots, tables).
Ask AI questions about your dataset.

🔮 Future Enhancements
Add advanced statistical analysis
Export AI-generated insights as PDF/Excel
Integrate with cloud storage (Google Drive / AWS S3)
Multi-file comparison feature

🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss your ideas.

📜 License
This project is licensed under the MIT License.

👨‍💻 Author
Mohd Mufeed
🎓 BTech in Information Technology
📍 Lucknow, India
📧 mufeedmohammad632@gmail.com






