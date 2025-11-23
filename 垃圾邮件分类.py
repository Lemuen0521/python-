# 提取邮件特征，对邮件进行分类
# 数据准备
# (一)提取监督信息
# 1.读取原始监督信息
with open('D:/垃圾邮件分类/trec06c/full/index') as file:
    y = [k.split()[0] for k in file.readlines()]
print(len(y))

# 2.用sklearn提供的工具"LabelEncoder"将原始监督信息转化为数值表示
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
y_encode = labelEncoder.fit_transform(y)

# (二)提取邮件文本特征向量
# 1.读取所有邮件文本的路径
import os
def getFilePathList2(rootDir):
    filePath_list = []
    for walk in os.walk(rootDir):
        part_filePath_list = [os.path.join(walk[0], file) for file in walk[2]]
        filePath_list.extend(part_filePath_list)
    return filePath_list
filePath_list = getFilePathList2('D:/垃圾邮件分类/trec06c/data')

# 2.加载邮件正文文本
mailContent_list = []
for filePath in filePath_list:
    with open(filePath, errors='ignore') as file:
        file_str = file.read()
        mailContent = file_str.split('\n\n', maxsplit=1)[1] #只保留正文部分的内容
        mailContent_list.append(mailContent)
print(mailContent_list[1])

# 3.文本数据预处理
# (3.1) 去除文本中多余空格
import re
mailContent_list = [re.sub(r'\s+', ' ', k) for k in mailContent_list]
# (3.2) 加载停用词表（中文停用词，例如"的"、"地"、"得"等，对文字意义影响不大的字或词。）
with open('D:/垃圾邮件分类/stopwords.txt', encoding='utf8') as file:
    file_str = file.read()
    stopword_list = file_str.split('\n')
    stopword_set = set(stopword_list)
# (3.3) 中文分词
import jieba
import time
cutWords_list = []
startTime = time.time()
for mail in mailContent_list:
    cutWords = [k for k in jieba.lcut(mail) if k not in stopword_set]
    cutWords_str = ' '.join(cutWords)
    cutWords_list.append(cutWords_str)
# （3.4）提取邮件文本的tf-idf特征向量
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df=100, max_df=0.25)
X = tfidf.fit_transform(cutWords_list)
print(f"特征矩阵形状: {X.shape}")

# 模型训练——训练逻辑回归模型
# （1）训练数据和测试数据划分
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y_encode, test_size=0.2, random_state=42)

# （2）模型训练与测试
from sklearn.linear_model import LogisticRegressionCV
logistic_model = LogisticRegressionCV()#初始化一个逻辑回归模型
logistic_model.fit(train_X, train_y) #调用sklearn的模型训练程序

# 修复：分开计算准确率和输出
accuracy = logistic_model.score(test_X, test_y)  # 先计算准确率
accuracy_rounded = round(accuracy, 4)  # 再四舍五入
print(f"模型准确率: {accuracy_rounded}")

# （3）模型保存
import pickle
#将模型参数、数值类别对应的真实标签保存在文件allModel.pickle中
with open('allModel.pickle', 'wb') as file:
    save = {
        'labelEncoder' : labelEncoder,
        'tfidfVectorizer' : tfidf,
        'logistic_model' : logistic_model
    }
    pickle.dump(save, file)

# （4）加载并使用训练好的模型
import pickle
#加载训练好的模型
with open('allModel.pickle', 'rb') as file:
    allModel = pickle.load(file)
    labelEncoder = allModel['labelEncoder']
    tfidfVectorizer = allModel['tfidfVectorizer']
    logistic_model = allModel['logistic_model']
#使用训练好的模型，预测数据X的标签
predict_y = logistic_model.predict(X)

# 模型评估（自己加的）
# 模型在测试集上的准确率
accuracy = logistic_model.score(test_X, test_y)
accuracy_rounded = round(accuracy, 4)
print(f"模型准确率: {accuracy_rounded}")

# 预测能力
# 可以对新的邮件进行分类预测
def predict_new_email(email_content):
    # 文本预处理
    # 修复：在正则表达式前添加 r
    cleaned_content = re.sub(r'\s+', ' ', email_content)
    cut_words = [k for k in jieba.lcut(cleaned_content) if k not in stopword_set]
    # 修复：将分词列表转换为字符串
    cut_words_str = ' '.join(cut_words)

    # 特征提取
    email_tfidf = tfidfVectorizer.transform([cut_words_str])

    # 预测
    prediction_encoded = logistic_model.predict(email_tfidf)
    prediction = labelEncoder.inverse_transform(prediction_encoded)

    return prediction[0]  # 返回 'spam' 或 'ham'

# 测试预测功能
print("\n测试预测功能:")
test_emails = [
    "免费获得iPhone！点击链接立即领取大奖！",
    "您好，关于明天的会议安排，请查收附件。",
    "赚钱机会！投资1000元，日赚500元！"
]

for i, email in enumerate(test_emails):
    result = predict_new_email(email)
    print(f"测试邮件 {i+1}: {result}")
    print(f"内容: {email[:50]}...")
    print()

# 添加更详细的模型评估
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

print("\n=== 详细模型评估 ===")
test_predictions = logistic_model.predict(test_X)
print("分类报告:")
print(classification_report(test_y, test_predictions, target_names=labelEncoder.classes_))

# 计算混淆矩阵
cm = confusion_matrix(test_y, test_predictions)
print("混淆矩阵数值:")
print(cm)

# 绘制混淆矩阵热力图
plt.figure(figsize=(8, 6))
sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labelEncoder.classes_,
            yticklabels=labelEncoder.classes_,
            cbar_kws={'label': '样本数量'})

plt.title('垃圾邮件分类混淆矩阵热力图', fontsize=14, fontweight='bold')
plt.xlabel('预测标签', fontsize=12)
plt.ylabel('真实标签', fontsize=12)
plt.tight_layout()

# 保存图片
plt.savefig('confusion_matrix_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

