import pandas as pd
from sklearn.model_selection import train_test_split

label_list = ['news_headline','article','확진자수','완치자수','사망여부','집단감염','백신관련','방역지침','경제지원','마스크','국제기구','병원관련']

if __name__ == '__main__':
    data = pd.read_csv('detail_labeling.csv')
    data = data[label_list].dropna()
    
    train, test = train_test_split(data, shuffle=True, train_size=0.9)

    print(train)
    print(test)
    train.to_csv('train.csv', index=False)
    test.to_csv('test.csv', index=False)