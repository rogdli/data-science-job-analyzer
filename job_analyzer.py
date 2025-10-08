from flask import Flask, jsonify
import pandas as pd
from collections import Counter
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# función para limpiar salarios (vienen en formato string, con símbolos de euros y algunos con rangos)
def clean_salary(salary_str):
    if pd.isna(salary_str):
        return None
    if ' - ' in str(salary_str):
        # Si es rango, tomar el promedio
        low, high = str(salary_str).replace('€', '').replace(',', '').split(' - ')
        return (float(low) + float(high)) / 2
    else:
        # Si es valor único
        return float(str(salary_str).replace('€', '').replace(',', ''))

# cargar y limpiar datos
df = pd.read_csv('datasets/data_science_job_posts_2025.csv')
df['salary'] = df['salary'].apply(clean_salary)

@app.route("/top_skills")
# buscar habilidades más requeridas
def top_skills():
    skills = df['skills']
    skills_counter = Counter()
    for sublist in skills.dropna().str.split(','):
        skills_counter.update([skill.strip() for skill in sublist])
    return jsonify(skills_counter.most_common(10))

@app.route("/avg_salary_by_level")
# salario promedio por nivel de seniority
def avg_salary_by_level():
    salary_by_level = df.dropna(subset=['salary']).groupby('seniority_level')['salary'].mean().to_dict()
    return jsonify(salary_by_level)

@app.route("/most_wanted_jobs")
# trabajos más buscados
def most_wanted_jobs():
    wanted_jobs = df['job_title'].value_counts().head(10).to_dict()
    return jsonify(wanted_jobs)

@app.route("/top_companies")
# compañías con más ofertas
def top_companies():
    top_companies = df['company'].value_counts().head(10).to_dict()
    return jsonify(top_companies)

@app.route("/avg_salary_by_technology")
# salario promedio por tecnología
def avg_salary_by_technology():
    df_expanded = df.dropna(subset=['salary', 'skills'])['skills'].str.split(',', expand=True).stack().str.strip()
    df_expanded = df_expanded.to_frame('skill').reset_index(level=1, drop=True)
    df_expanded['salary'] = df.dropna(subset=['salary', 'skills'])['salary']
    tech_salary = df_expanded.groupby('skill')['salary'].mean().sort_values(ascending=False).head(10).to_dict()
    return jsonify(tech_salary)

@app.route("/jobs_by_location")
# trabajos por ubicación
def jobs_by_location():
    jobs_location = df['location'].dropna().value_counts().head(10).to_dict()
    return jsonify(jobs_location)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

