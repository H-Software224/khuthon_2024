from nltk.tokenize import RegexpTokenizer

sentence = "한글 자연어 처리(Natural Language Processing), 그렇게 어렵지 않네요.ㅋㅋㅋ"

tokenizer = RegexpTokenizer("[\s]+", gaps=True) 
                # 공백으로 구분, gaps 정규표현식을 토큰으로 나눌 것인지 여부
tokens = tokenizer.tokenize(sentence)
print(tokens)