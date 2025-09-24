# WasteAI: Smart Waste Management System

WasteAI is an AI-powered web application for waste classification, reporting, and gamification.  
It uses a deep learning model (`waste_classifier.h5`) for classifying waste images and integrates
gamified features (points, leaderboards, challenges) to encourage students and communities to manage waste responsibly.

---

## 🚀 Features
- AI Waste Classification (TensorFlow CNN)
- Waste Reporting System
- Gamification: Points, Challenges, Leaderboards
- User Authentication & Roles (User/Admin)
- School & Bin Management
- Create fun challenges within community, and earn points from it. 

---

## 🛠 Tech Stack
- **Backend:** Flask (Python)
- **Database:** SQLite + SQLAlchemy
- **AI Model:** TensorFlow + NumPy + Pillow
- **Frontend:** HTML, CSS, JavaScript
- **Auth & Security:** Flask-Login, Werkzeug password hashing

---

## 📦 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/CJ-PRO-bot/WasteAI-poc.git
   cd WasteAI
Create a virtual environment & activate it:

bash
Copy code
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the app:

bash
Copy code
python app.py
Open in browser:

cpp
Copy code
http://127.0.0.1:5000/
📊 Demo & Screenshots
[Prototype Video Demo](https://drive.google.com/file/d/13h08xbyYXeMXefhzRKdSMVvdhP4rwAnI/view?usp=sharing)

[System Diagram](https://drive.google.com/file/d/1C0EHlkGXnKey1lcv4o9x0PdbawZckzfw/view?usp=sharing)