import jieba
import jieba.posseg as jbpos
import jieba.analyse as jbal

'''
词性说明:
a:形容词
d:副词
i:成语
m:数词
n:名词
nr:人名
ns:地名
nt:机构团体
nz:其他专有名词
t:时间
v:动词
x:标点符号
f:方位词
un:未知
'''

if __name__ == '__main__':
    string1 = "国内掀起了大数据、云计算的热潮。"

    # 全模式
    w1 = jieba.cut(string1, cut_all=True)
    print(w1)
    # for i in w1:
    #     print(i)

    # 精准模式，默认是精准模式
    w2 = jieba.cut(string1)
    # for i in w2:
    #     print(i)
    # print("<----------->")

    # 搜索引擎模式
    w3 = jieba.cut_for_search(string1)
    # for i in w3:
    #     print(i)
    # print("<----------->")

    # 词性标注
    w4 = jbpos.cut(string1)
    # for i in w4:
    #     print(i.word + "--" + i.flag)
    # print("<----------->")

    # 词典加载
    # jieba.load_userdict("dict2.txt")
    string2 = "国内掀起了大数据、云计算的热潮。仙鹤门地区。"
    #word 词语，flag 词性
    w5 = jbpos.cut(string2)
    for i in w5:
        print(i.word + "--" + i.flag)
    print("<----------->\n")

    # 更改词频-单个词
    jieba.suggest_freq("大数据", True)
    jieba.suggest_freq("云计算", True)
    w6 = jbpos.cut(string2)
    for i in w6:
        print(i.word + "--" + i.flag)
    print("<----------->\n")

    # 动态修改词典 删除词 del_word
    jieba.add_word("仙鹤门")
    w7 = jbpos.cut(string2)
    for i in w7:
        print(i.word + "--" + i.flag)
    print("<----------->\n")

    # 提取关键词 第二个参数控制提取参数个数
    w8 = jbal.extract_tags(string2, 5)
    print(w8)