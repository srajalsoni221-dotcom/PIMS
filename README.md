Make the folder first : Predictive_Project_Monitoring

Installation and Run: 
git clone <your-github-repository-url>
cd Predictive_Project_Monitoring

python -m venv .venv
.venv\Scripts\activate

pip install streamlit pandas numpy plotly requests fastapi uvicorn pydantic

Run the fastApi in backend :
python api.py

Open another terminal, activate .venv, then run Streamlit:
streamlit run app.py

Login:
Username: admin
Password: pims123
