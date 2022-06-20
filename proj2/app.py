from flask import Flask, render_template, request
from back2 import back2

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def root():
    if request.method == "GET":
        return render_template('index.html')
    elif request.method == "POST":
        l_distro = request.form["l_distro"].lower()
        part = request.form["part"]
        assert l_distro in ["pareto", "expo"]
        try:
            back2(l_distro, part)
            return render_template('result.html')
        except NotImplementedError :
            return render_template('not_impl.html')



if __name__ == "__main__":
    app.run()
