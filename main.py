from sklearn.model_selection import train_test_split
from data_preprocessing import *
from inspection import *
from train import *
from models import *
from file_combine import *
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# nltk.download('wordnet')
# nltk.download('stopwords')

text_path = []
# Without personal features:
# for i in range(12):
#     base_path = "texts/2012/text_2012-"
#     if i < 9:
#         path = base_path + "0%d_noTM.csv"%(i+1)
#     else:
#         path = base_path + "%d_noTM.csv"%(i+1)
#     text_path.append(path)


# With personal features:
for i in range(12):
    base_path = "texts/2012/text_2012-"
    if i < 9:
        path = base_path + "0%d_noTM_personal.csv"%(i+1)
    else:
        path = base_path + "%d_noTM_personal.csv"%(i+1)
    text_path.append(path)



# Split training and test sets:
def train_test(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.15, random_state=0)
    return X_train, X_test, y_train, y_test


def labels(df):
    label_encoder = LabelEncoder()
    label_encoder.fit(df.Emoji)
    df['emoji_id'] = label_encoder.transform(df.Emoji)


def result(wfile, X, y):
    X_train, X_test, y_train, y_test = train_test(X, y)
    model_list = ["Random Forest", "Logistic Regression", "SVM", "KNN"]
    for cl_model in model_list:
        model = train_models(cl_model, X_train, y_train)
        predictions,accuracy = make_prediction(model, X_train, y_train)
        wfile.write("The accuracy of the model %s for training is: %f\n"%(cl_model, accuracy))
        predictions,accuracy = make_prediction(model, X_test, y_test)
        wfile.write("The accuracy of the model %s for testing is: %f\n"%(cl_model, accuracy))


def main(path):
    
    df = read_files(path)
    feature_inspect(df)
    word2vec_pre(df)
    dic = LDA_pre(df)
    top_word(df)
    norm(df)
    print(df.head(5))

    # LDA model:
    LDAmodel = LDA_train(df, dic)
    df['LDA_features'] = list(map(lambda text: LDA_features(LDAmodel, text), df.Bow))
    LDA_example_top_topic(df, LDAmodel)

    # W2V model:
    W2Vmodel = Word2vec_train(df)
    df['W2V_features'] = list(map(lambda text: W2V_features(W2Vmodel, text), df.Tokenized_Sentence))

    labels(df)
    LDA_X = np.array(list(map(np.array, df.LDA_features)))
    W2V_X = np.array(list(map(np.array, df.W2V_features)))

    # Without personal features:
    # X = np.append(LDA_X, W2V_X, axis = 1)
    # print(X)

    # With personal features:
    X = np.append(LDA_X, W2V_X, axis = 1)
    personal_X_C = np.array(list(map(np.array, list(df.Ckarma_scaled)))).reshape(X.shape[0],1)
    personal_X_L = np.array(list(map(np.array, list(df.Lkarma_scaled)))).reshape(X.shape[0],1)
    personal = np.append(personal_X_C, personal_X_L, axis = 1)
    X = np.append(X, personal, axis = 1)
    print(X)
    personality = np.array(list(map(np.array, list(zip(df.Personality_IE,df.Personality_NS,df.Personality_TF,df.Personality_PJ))))).reshape(X.shape[0],4)
    X = np.append(X, personality, axis = 1)
    print(X)
    y = df.emoji_id

    # with open('training_result_2012_month.txt', 'a') as wfile:
    #     wfile.write("See the results for %s: \n"%path)
    #     string1 = "Training results by combining LDA and W2V methods:\n"
    #     wfile.write(string1)
    #     result(wfile, X, y)
    #     wfile.write("**************************************************\n")
    # with open('training_result_2012_year.txt', 'a') as wfile:
    #     wfile.write("See the results for 2012: \n")
    #     string1 = "Training results by combining LDA and W2V methods:\n"
    #     wfile.write(string1)
    #     result(wfile, X, y)
    # with open('training_result_2012_year_noTM.txt', 'a') as wfile:
    #     wfile.write("See the results for 2012 without TM mark: \n")
    #     string1 = "Training results by combining LDA and W2V methods:\n"
    #     wfile.write(string1)
    #     result(wfile, X, y)
    with open('training_result_2012_year_noTM_personal.txt', 'a') as wfile:
        wfile.write("See the results for 2012 without TM mark and with personal features: \n")
        string1 = "Training results by combining LDA and W2V methods:\n"
        wfile.write(string1)
        result(wfile, X, y)


if __name__ == "__main__":
    # with open('training_result_2013_month.txt', 'w') as wfile:
    #     pass
    # For monthly data:
    # for path in text_path:
    #     main(path)
    # Aggregate all data for a year:
    Wfile = "texts/2012/text_2012_noTM_personal.csv"
    df = combine(text_path)
    df.to_csv(Wfile, encoding='utf-8', index=False, header= True)
    main(Wfile)