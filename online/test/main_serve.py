import requests


def test_main_serve():
    # 设置服务请求的地址URL
    url = "http://0.0.0.0:5000/main_serve/"
    data = {"text1": "人生该如何起头", "text2": "改变要如何起手"}
    res = requests.post(url, data=data)

    assert res.text == '人生该如何起头改变要如何起手'
