from flask import Flask
from flask import request

app = Flask(__name__)


@app.route('/main_serve/', methods=['POST'])
def recognition():
    # 首先接收数据
    text_1 = request.form['text1']
    text_2 = request.form['text2']
    return text_1 + text_2
