from flask import Flask, render_template, request
import pandas as pd
app = Flask(__name__)
try:
    df = pd.read_csv('laptop_scrap_data.csv')
except FileNotFoundError:
    df = pd.DataFrame(columns=['Company', 'TypeName', 'Cpu', 'Ram_GB', 'Price'])
@app.route('/')
def index():
    data = df.head(50).to_dict(orient='records')
    return render_template('index.html', laptops=data, columns=df.columns)
@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query', '').lower()
    if query:
        mask = df.apply(lambda row: row.astype(str).str.lower().str.contains(query).any(), axis=1)
        filtered_df = df[mask]
    else:
        filtered_df = df.head(50)
    data = filtered_df.to_dict(orient='records')
    return render_template('index.html', laptops=data, columns=df.columns)
if __name__ == '__main__':
    app.run(debug=True)