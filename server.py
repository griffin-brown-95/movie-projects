from flask import Flask, request, render_template
from movie_selector import get_recommendations_table
from waitress import serve
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations_table = None
    movie_title = None
    if request.method == 'POST':
        movie_title = request.form['title']
        recommendations_table = get_recommendations_table(movie_title)
    return render_template('home.html', movie_title=movie_title, recommendations_table=recommendations_table)

""" if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True) """

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    serve(app, host="0.0.0.0", port=port)