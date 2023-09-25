import requests


def test_main_serve():
    # 设置服务请求的地址URL
    url = "http://0.0.0.0:5000/main_serve/"
    data = {"uid": "12", "text": "改变要如何起手"}
    res = requests.post(url, data=data)
    print(res.text)

    # assert res.text == '人生该如何起头改变要如何起手'


test_main_serve()
