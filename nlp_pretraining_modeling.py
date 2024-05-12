# LLM(거대 생성형 언어 모델 (Attention 등 고성능 모델 이용))
# OPENAI(GPT-3.5-turbo)(GPT3.5)
# 텍스트 키워드 추출
from openai import OpenAI
import os
import pandas as pd
# gpt_5_client = OpenAI(api_key='open')


# 텍스트 데이터 다운로드 한 다음에 유의
# completion = gpt_5_client.chat.completions.create(
#   model="gpt-3.5-turbo",
#   messages=[
#     {"role": "system", "content": "무엇을 도와드릴까요?"},
#     {"role": "user", "content": "khuthon을 잘하려면 어떻게 해야 돼?"}
#   ]
# )

# print(completion.choices[0].message)

# embedding_client = OpenAI(api_key='your_openai_key')

# response = embedding_client.embeddings.create(
#     input="Your text string goes here",
#     model="text-embedding-3-small"
# )

# print(response.data[0].embedding)

# class encoding:
#   #S, A+, A 등의 level에 대응하는 one-hot vector return
#   #직접적으로 사용 X
#   def match(level):
#     level_matching = {'S':[1,0,0,0,0,0,0], 'A+':[0,1,0,0,0,0,0], 'A':[0,0,1,0,0,0,0],
#                       'B+':[0,0,0,1,0,0,0], 'B':[0,0,0,0,1,0,0], 'C':[0,0,0,0,0,1,0], 'D' : [0,0,0,0,0,0,1]}

#     return level_matching[level]

#   #ex: (S, A+, D, C) -> 각 level을 one-hot vector로 표현하여, 행렬 return
#   #직접적으로 사용 O
#   def encode(level):
#     encoding = [match[level[i]] for i in range(len(level))]
#     return encoding
# print(len(" 이 글에 대해서 키워드 추출해줘."))
# class embedding:
#   def __init__(self):
#     pass

#   def embed(self, client, word):
#     response = client.embeddings.create(input = "%s" %word,  model="text-embedding-3-small")

#     #embedding_vector 추출
#     embedding_vector = response.data[0].embedding

#     return embedding_vector
  
NUMBER = 5 #분야 별 키워드 개수

# text 파일 열기
# newfile.py
# f = open("./text_data/GS리테일.txt", 'r', encoding='UTF8')
# contents = f.read()
# print(len(contents))
# # print(contents) 
# f.close()

# from openai import OpenAI

# gpt_5_client = OpenAI(api_key='your_openai_key')


# # 텍스트 데이터 다운로드 한 다음에 유의
# for i in (len())
# completion = gpt_5_client.chat.completions.create(
#   model="gpt-3.5-turbo",
#   messages=[ {"role": "system", "content": "무엇을 도와드릴까요?"}, {"role": "user", "content": f"{contents} 이 글에 대해서 키워드 추출해줘."}]
# )
# # 답변 설명
# print(completion.choices[0].message)

# messages = []
# system1 = {"role":"system", "content": "무엇을 도와 드릴까요"}
# messages.append()
#  이 글에 대해서 키워드 추출해줘.
# pdf마다 6000토큰 달성
from nltk.tokenize import RegexpTokenizer
message_6000_list = []
file_list = os.listdir('./text_data')
for i in file_list:
    f = open(f"./text_data/{i}", 'r', encoding='UTF8')
    contents = f.read()
    print(len(contents))
    f.close()
    number_count = 0
    txt_sample = []
    while contents != '':
        if len(contents) >= 5500:
            txt_sample.append(i[number_count:number_count+60])
            contents = contents[number_count+5500:]
        else:
            txt_sample.append(contents)
            contents = ''
        number_count += 5500
    message_6000_list.append(txt_sample)


# 텍스트 데이터 다운로드 한 다음에 유의
total_keyword_list=[]
for message in message_6000_list:
    key_word_list = []
    for i in message:
        gpt_5_client = OpenAI(api_key='your_openai_key')

        # 텍스트 데이터 다운로드 한 다음에 유의
        completion = gpt_5_client.chat.completions.create(
          model="gpt-3.5-turbo",
          messages=[
            {"role": "system", "content": "무엇을 도와드릴까요?"},
            {"role": "user", "content": f"{i} 이 내용을 중요한 명사로 추출해줘."}
          ]
        )
        print(completion.choices[0].message.content)
        key_word_list.append(completion.choices[0].message.content)
    total_keyword_list.append(key_word_list)
print(total_keyword_list)

# 한국 형태소 클래스 추출
keyword_lists = []
tokenizer = RegexpTokenizer("[\s]+", gaps=True)
for total_keyword in total_keyword_list:
    for keyword in total_keyword:
        tokens = tokenizer.tokenize(keyword)
        print(tokens)
        for token in tokens:
            keyword_lists.append(token)




# csv파일로 만들어서 데이터 형식으로 만들기
keyword_dataframe = pd.DataFrame({'keyword':keyword_lists})
keyword_dataframe.to_csv('keyword.csv')


# Embedding = embedding
# Encoding = encoding
# client = OpenAI(api_key='your_openai_key')

# X, y = [], []

# data = ['행복', '밤샘', '기력', '없어', '죽겠', '다', '죽', '겠','다', '힘','들','다','키워드', '분야', '별', '키워드', '집', '가고', '싶다', '집', [1,0,0,0,0,0,0], [1,0,0,0,0,0,0] , [1,0,0,0,0,0,0], [1,0,0,0,0,0,0]]

# for i, row in enumerate(data):
#   X[i] = [Embedding().embed(client, row[j]) for j in range(20)] #word embedding
#   y[i] = [Encoding().encode(row[j]) for j in range(20, len(row))] #level encoding


# from torch.utils.data import TensorDataset, DataLoader
# from sklearn.model_selection import train_test_split
# import torch

# X = torch.stack([torch.tensor(sample) for sample in X])
# y = torch.stack([torch.tensor(sample) for sample in y])

# # shuffle & split

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# #DataLoader

# BATCH_SIZE = 50

# train_dataset = TensorDataset(X_train, y_train)
# test_dataset = TensorDataset(X_test, y_test)
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# import torch.nn as nn

# class MLP(nn.module):
#   def __init__(self, num_features)
#     super(MLP, self).__init__()

#     self.linear_1 = nn.Linear(num_features*5, 4096)
#     self.linear_2 = nn.Linear(4096, 1024)
#     self.linear_3 = nn.Linear(1024, 256)
#     self.linear_4 = nn.Linear(256, 64)
#     self.linear_5 = nn.Linear(64, 7)

#   def forward(self, x):
#     x = torch.ReLU(self.linear_1(x))
#     x = torch.ReLU(self.linear_2(x))
#     x = torch.ReLU(self.linear_3(x))
#     x = torch.ReLU(self.linear_4(x))
#     x = torch.ReLU(self.linear_5(x))

#     logits = self.linear_out(x)
#     probas = self.softmax(logits)

#     #CrossEntropyLoss에 Softmax가 내장되어 있기 때문에, Loss 계산 시 logits 이용

#     return logits, probas

# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# model = MLP(num_features = 768)
# optimizer =torch.optim.SGD(model.parameters(), lr = 0.001,momentum = 0.9, weight_decay = 0.0005)
# Loss = nn.CrossEntropyLoss()

# EPOCH = 100

# for epoch in range(EPOCH):
#   model.train()

#   predict = torch.zeros(BATCH_SIZE, 4, 7)

#   for i, (X, y) in enumerate(train_loader):
#     model.train()
#     X = X.to(DEVICE)

#     for j in range(0, 4):
#       predict[:, j, :], _ = model(X[5*j, 5*(j+1)])

#       loss = Loss(y, predict)

#       optimizer.zero_grad()
#       loss.backward()
#       optimizer.step()

#       if not i % 40:
#         print (f'Epoch: {epoch+1:03d}/{EPOCH:03d} | '
#                f'Batch {i:03d}/{len(train_loader):03d} |'
#                f' Cost: {loss:.4f}')
#     model.eval()

#     train_acc, train_loss = compute_accuracy_and_loss(model, train_loader, device=DEVICE)
#     test_acc, test_loss = compute_accuracy_and_loss(model, test_loader, device=DEVICE)
#     print(f'Epoch: {epoch+1:03d}/{EPOCH:03d} Train Acc.: {train_acc:.2f}%'
#           f' | Test Acc.: {test_acc:.2f}%')
