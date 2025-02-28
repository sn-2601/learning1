from flask import Flask, render_template
from unsupervised import unsupervised_clustering, plot_2d_densityf, plot_2d_densitym, plot_2d_densityf_1, plot_2d_densitym_1

app = Flask(__name__)

@app.route('/')
def home():
    # Run clustering
    df = unsupervised_clustering()

    # Generate density plots
    plot_2d_densityf(df)
    plot_2d_densitym(df) 
    plot_2d_densityf_1(df)
    plot_2d_densitym_1(df)

    # Render homepage
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
