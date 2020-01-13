import praw
from MBPT import *


def get_karma(username, reddit):
    redditor = reddit.redditor(username)
    comment_karma = redditor.comment_karma
    link_karma = redditor.link_karma
    return comment_karma, link_karma


def get_submissions(username, reddit):
    redditor = reddit.redditor(username)
    submissions = redditor.submissions.top('all')
    results = []
    for s in submissions:
        if s.selftext != '':
            results.append(s.selftext)
    print("There are %d submissions from this user."%len(results))
    return results    


def get_comments(username, reddit):
    redditor = reddit.redditor(username)
    comments = redditor.comments.top('all')
    results = []
    for c in comments:
        if c.body != '':
            results.append(c.body)
    print("There are %d comments from this user."%len(results))
    return results 


# print(get_karma("chuggo_tuggans"))

def get_personality(username, reddit):

    # Concat all the posts of this user:
    submissions = get_submissions(username, reddit)
    comments = get_comments(username, reddit)
    # print(submissions)
    # print("************************************")
    # print(comments)
    post_list = []
    for s in submissions:
        post_list.append(s)
    for c in comments:
        post_list.append(c)
    posts = "".join(post_list)
    posts = posts.replace('\n','')
    print(posts)
    print(type(posts))
    my_posts = posts

    mydata = pd.DataFrame(data={'type': ['INFJ'], 'posts': [my_posts]})

    my_posts, dummy = preprocessing(mydata)

    my_X_cnt = cntizer.transform(my_posts)
    my_X_tfidf =  tfizer.transform(my_X_cnt).toarray()

    result = []
    # Let's train type indicator individually
    for l in range(len(type_indicators)):
        print("%s ..." % (type_indicators[l]))
        
        model = pickle.load(open('personal_model_%s.sav'%str(l), 'rb'))
        # make predictions for my  data
        y_pred = model.predict(my_X_tfidf)
        result.append(y_pred[0])
        # print("* %s prediction: %s" % (type_indicators[l], y_pred))
    print("The result is: ", translate_back(result))
    personality = result
    
    return personality
    


