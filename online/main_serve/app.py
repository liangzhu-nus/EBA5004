from flask import Flask
from flask import request
import requests
import redis
import json
# from unit import unit_chat  # 导入已经编写好的Unit API文件
from neo4j import GraphDatabase  # 导入操作neo4j数据库的工具
from config import NEO4J_CONFIG, REDIS_CONFIG, TIMEOUT, reply_path
from config import model_serve_url, ex_time


app = Flask(__name__)
pool = redis.ConnectionPool(**REDIS_CONFIG)

# 初始化neo4j的驱动对象
_driver = GraphDatabase.driver(**NEO4J_CONFIG)


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
        """_summary_

        Args:
            uid (_type_): unique user identification
            text (_type_): current input text of user
            r (_type_): a instance of redis object
            reply (_type_): constant dialogue template
        """
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

    # 编码首句请求的代码函数
    def first_sentence(self):
        """user first request, don't need to call bert model serve, just do neo4j search"""

        diseases_list = query_neo4j(self.text)

        if not diseases_list:
            return "sorry, i can't help you with your describtion"

        self.r.hset(str(self.uid), "previous_d", str(diseases_list))
        # 将查询回来的结果存储进redis, 并且做为下一次访问的"上一条语句"previous
        self.r.hset(str(self.uid), "previous_d", str(diseases_list))
        # 设置数据库的过期时间
        self.r.expire(str(self.uid), ex_time)
        # 将列表转换为字符串, 添加进规则对话模板中返回给用户
        res = ",".join(diseases_list)
        # 此处打印信息, 说明neo4j查询后有结果并且非空, 接下来将使用规则模板进行对话生成
        print("使用规则对话生成模板进行返回对话的生成!")
        # TODO:
        return self.reply["2"] % res

    # 编写非首句处理函数, 该用户不是第一句问话
    def non_first_sentence(self, previous):
        '''
        previous: 代表该用户当前语句的上一句文本信息
        '''
        try:
            data = {"text1": previous, "text2": self.text}
            # 直接向语句服务模型发送请求
            result = requests.post(model_serve_url, data=data, timeout=TIMEOUT)
            # 如果回复为空, 说明服务暂时不提供信息, 转去百度机器人回复
            if not result.text:
                return "sorry, i can't help you with your describtion"
        except Exception as e:
            return "sorry, i can't help you with your describtion"

        # 句子相关模型服务请求成功且不为空
        diseases_list = query_neo4j(self.text)
        if not diseases_list:
            # 判断如果结果为空, 继续用百度机器人回复
            return "sorry, i can't help you with your describtion"

        # 如果结果不是空, 从redis中获取上一次已经回复给用户的疾病名称
        old_disease = self.r.hget(str(self.uid), "previous_d")

        # 如果曾经回复过用户若干疾病名称, 将新查询的疾病和已经回复的疾病做并集, 再次存储
        # 新查询的疾病, 要和曾经回复过的疾病做差集, 这个差集再次回复给用户
        if old_disease:
            # new_disease是本次需要存储进redis数据库的疾病, 做并集得来
            new_disease = list(set(diseases_list) | set(eval(old_disease)))
            # 返回给用户的疾病res, 是本次查询结果和曾经的回复结果之间的差集
            res = list(set(diseases_list) - set(eval(old_disease)))
        else:
            # 如果曾经没有给该用户的回复疾病, 则存储的数据和返回给用户的数据相同, 都是从neo4j数据库查询返回的结果
            res = new_disease = list(set(diseases_list))

        # 将new_disease存储进redis数据库中, 同时覆盖掉之前的old_disease
        self.r.hset(str(self.uid), "previous_d", str(new_disease))
        # 设置redis数据的过期时间
        self.r.expire(str(self.uid), ex_time)
        # 此处打印信息, 说明neo4j查询后已经处理完了redis任务, 开始使用规则对话模板
        # 将列表转化为字符串, 添加进规则对话模板中返回给用户
        if not res:
            return self.reply["4"]
        else:
            res = ",".join(res)
            return self.reply["2"] % res


# 查询neo4j图数据的函数
def query_neo4j(text):
    ''''
    功能: 根据用户对话文本中可能存在的疾病症状, 来查询图数据库, 返回对应的疾病名称
    text: 用户输入的文本语句
    return: 用户描述的症状所对应的的疾病名称列表
    '''
    # 开启一个会话session来操作图数据库
    with _driver.session() as session:
        # 构建查询的cypher语句, 匹配句子中存在的所有症状节点
        # 保存这些临时的节点, 并通过关系dis_to_sym进行对应疾病名称的查找, 返回找到的疾病名称列表
        cypher = "MATCH(a:Symptom) WHERE(%r contains a.name) WITH \
                 a MATCH(a)-[r:dis_to_sym]-(b:Disease) RETURN b.name LIMIT 5" \
                 % text
        # 通过会话session来运行cypher语句
        record = session.run(cypher)
        # 从record中读取真正的疾病名称信息, 并封装成List返回
        result = list(map(lambda x: x[0], record))
    return result
