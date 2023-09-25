from flask import Flask
from flask import request
import redis
import json
from config import REDIS_CONFIG
from config import reply_path


app = Flask(__name__)
pool = redis.ConnectionPool(**REDIS_CONFIG)


@app.route('/main_serve/', methods=['POST'])
def main_serve():
    # 此处打印信息, 说明werobot服务成功的发送了请求
    print("已经进入主要逻辑服务, werobot服务正常运行!")

    # 首先接收数据
    uid = request.form['uid']
    text = request.form['text']

    # 从redis连接池中获得一个活跃的连接
    r = redis.StrictRedis(connection_pool=pool)

    # 获取该用户上一次说的话(注意: 可能为空)
    previous = r.hget(str(uid), "previous")
    # 将当前输入的text存入redis, 作为下一次访问时候的"上一句话"
    r.hset(str(uid), "previous", text)

    # 此处打印信息, 说明redis能够正常读取数据和写入数据
    print("已经完成了初次会话管理, redis运行正常!")
    
    reply = json.load(open(reply_path, "r"))

    # 实例化Handler类
    handler = Handler(uid, text, r, reply)

    # 如果上一句话存在, 调用非首句服务函数
    if previous:
        print("非首句调用")
        return handler.non_first_sentence(previous)
    # 如果上一句话不存在, 调用首句服务函数
    else:
        print("首句调用")
        return handler.first_sentence()


# 主要逻辑服务类Handler类
class Handler(object):
    def __init__(self, uid, text, r, reply):
        '''
        uid: 用户唯一标识uid
        text: 标识该用户本次输入的文本信息
        r: 代表redis数据库的一个链接对象
        reply: 规则对话模板加载到内存中的对象(字典对象)
        '''
        self.uid = uid
        self.text = text
        self.r = r
        self.reply = reply

    # 编写非首句处理函数, 该用户不是第一句问话
    def non_first_sentence(self, previous):
        '''
        previous: 代表该用户当前语句的上一句文本信息
        '''
        return 'non_sentence'

    # 编码首句请求的代码函数
    def first_sentence(self):
        return 'first_sentence'
