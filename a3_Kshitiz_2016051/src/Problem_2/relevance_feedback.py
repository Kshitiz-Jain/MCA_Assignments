import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


def load_gt(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    a = [(int(l.split()[0]), int(l.split()[2])) for l in lines]
    truth = []
    for i in range(30):
        truth.append([])
    for i in a:
        truth[i[0]-1].append(i[1]-1)
    return truth


def relevance_feedback(vec_docs, vec_queries, sim, n=10):
    rf_sim=np.array(sim)
    trueVals = load_gt("data/MED.REL")

    vec_queries=vec_queries.todense()


    alpha = 0.75
    beta = 0.15

    for k in range(3):
        for j in range(np.shape(vec_queries)[0]):
            ranked_documents = np.argsort(-rf_sim[:, j])[:n]

            relevant = []
            nonRelevant = []
            for i in range(np.shape(ranked_documents)[0]):
                if(ranked_documents[i] in trueVals[j]):
                    relevant.append(ranked_documents[i])
                else:
                    nonRelevant.append(ranked_documents[i])
            bestMatch = vec_docs[relevant,:].sum(axis=0)
            worstMatch = vec_docs[nonRelevant,:].sum(axis=0)

            if(len(relevant)!=0):
                bestMatch=bestMatch/len(relevant)
            else:
                bestMatch = np.zeros(np.shape(vec_queries[j]))
            if(len(nonRelevant)!=0):
                worstMatch=worstMatch/len(nonRelevant)
            else:
                worstMatch = np.zeros(np.shape(vec_queries[j]))

            vec_queries[j] += (alpha*bestMatch - beta*worstMatch)

        rf_sim = cosine_similarity(vec_docs, vec_queries)

    return rf_sim



def createTheasurus(dataMat):
    dataMat = normalize(np.transpose(dataMat))
    C = np.dot(dataMat,np.transpose(dataMat))
    return C


def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, n=10):
    rf_sim=np.array(sim)
    trueVals = load_gt("data/MED.REL")

    #Global Search
    thea = createTheasurus(vec_docs)

    vec_queries=vec_queries.todense()
    alpha = 0.75
    beta = 0.15

    for k in range(3):
        for j in range(np.shape(vec_queries)[0]):
            ranked_documents = np.argsort(-rf_sim[:, j])[:n]

            relevant = []
            nonRelevant = []
            for i in range(np.shape(ranked_documents)[0]):
                if(ranked_documents[i] in trueVals[j]):
                    relevant.append(ranked_documents[i])
                else:
                    nonRelevant.append(ranked_documents[i])
            bestMatch = vec_docs[relevant,:].sum(axis=0)
            worstMatch = vec_docs[nonRelevant,:].sum(axis=0)

            if(len(relevant)!=0):
                bestMatch=bestMatch/len(relevant)
            else:
                bestMatch = np.zeros(np.shape(vec_queries[j]))
            if(len(nonRelevant)!=0):
                worstMatch=worstMatch/len(nonRelevant)
            else:
                worstMatch = np.zeros(np.shape(vec_queries[j]))

            vec_queries[j] += (alpha*bestMatch - beta*worstMatch)


        #     #Query Extension
        # # for j in range(np.shape(vec_queries)[0]):
        #     query = np.ravel(vec_queries[j])
        #     idx = np.argsort(-query)[:n]

        #     for i in range(n):
        #         topIdx = idx[i]
        #         simTerms = np.ravel(thea[topIdx].toarray())
        #         topSimIdx = np.argsort(-simTerms)[:n]
        #         query[topSimIdx] = query[topIdx]
        #         vec_queries[j]=query

            query = np.ravel(vec_queries[j])
            topDocsIdx = trueVals[j]
            topDocs = vec_docs[topDocsIdx,:].todense()
            topWords = np.argsort(-topDocs, axis=1)[:,:10]
            for i in range(np.shape(topWords)[0]):
                temp = np.ravel(topWords[i])
                query[temp] += np.ravel(topDocs[i])[temp]

            # vec_queries[j] = np.ravel(normalize([query]))
            vec_queries[j] = query


        rf_sim = cosine_similarity(vec_docs, vec_queries)

    return rf_sim




                # #Local Search
                # rankDocs = np.argsort(-sim[:, j])[:2*n]
                # topDocs = vec_docs[rankDocs,:]
                # thea = createTheasurus(topDocs)









