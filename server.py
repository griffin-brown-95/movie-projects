from flask import Flask, request, render_template
from movie_selector import get_recommendations, movie_titles
from waitress import serve
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = None
    movie_title = None
    if request.method == 'POST':
        movie_title = request.form['title']
        recommendations = get_recommendations(movie_title)
    return render_template('home.html', movie_titles=movie_titles, movie_title=movie_title, recommendations=recommendations)

""" if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    serve(app, host="0.0.0.0", port=port) """

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)