import nltk
import itertools
import numpy as np
import streamlit as st
nltk.download('punkt')
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

st.title("Keyword Extraction ~ find the key topics addressed in a document")

# ---------------------------------------------------------------------------- #

def max_sum_similarity(doc_embedding, candidate_embeddings, kw_candidates, num_candidates, top_n):
    # find the distances between the document embedding and candidate embedding (for each of the possible 1-to-n gram candidates)
    # we want to minimize this distance to produce accurate keyphrases
    doc_and_candidate_distances = cosine_similarity(doc_embedding, candidate_embeddings)

    # find the distances between each of the candidate embeddings
    # we want to maximize this distance to produce diverse keyphrases
    inter_candidate_distances = cosine_similarity(candidate_embeddings, candidate_embeddings)


    # first, get the top 'nr_candidates' candidates that are closest to doc embedding
    # these are the shortlisted candidates
    candidate_indices = list(doc_and_candidate_distances.argsort()[0][-num_candidates:])
    candidates = [kw_candidates[idx] for idx in candidate_indices]

    # now, get the inter_candidate_distances between each of these shortlisted kw candidates
    inter_shortlistedcandidate_distances = inter_candidate_distances[np.ix_(candidate_indices, candidate_indices)]
    # explanation:
    # the above logic extracts from inter_candidate_distances only those
    # distances that are between our shortlisted candidates. for eg, if we had 1000 candidates
    # to begin with i.e. inter_candidate_distances.shape = (1000, 1000), and we shortlisted say 20
    # then we would take only the cells which carried the distance between these 20 candidates.


    # finally, take the shortlisted candidates least similar to each other to increase diversity
    # we do this by taking all combinations of the total 'nr_candidates' shortlisted candidates which
    # are top_n in length, and check which one has the lowest intercandidate similarity between the top_n candidates
    min_sim = np.inf
    best_combination = None

    # note that nr_candidates = num of shortlisted candidates. however, we may not be able to generate that many
    # so it could be that actual num of shortlisted candidates < nr_candidates
    # Hence, we take their count rather than directly specifying nr_candidates
    for combination in itertools.combinations(range(len(candidate_indices)), top_n): # equivalent to C(num of shortlisted candidates, top_n)
        # can also be termed as intra_combination_similarity
        inter_candidate_similarity = sum([inter_shortlistedcandidate_distances[i][j] for i in combination for j in combination])
        if inter_candidate_similarity < min_sim:
            best_combination = combination
            min_sim = inter_candidate_similarity

    # now, 'best_combination' is a list of the candidate keyphrases which...
        # i.  have maximum similarity with the doc embedding (for ACCURACY)
        # ii. have minimum similarity with other candidate keyphrases in the combination (for DIVERSITY)
    final_kws = [candidates[idx] for idx in best_combination]

    return final_kws


def encode_long_document(model, DOC):
    """
        Breaks up long document into individual paragraphs, derives embedding for each and mean-pools all to arrive at overall doc_embedding.
    """

    # Extract paragraphs and disregard any short ones or Table/Figure ones
    paras = DOC.split("\n")
    paras = [para for para in paras if (len(para) >= 55 and not (para.startswith("Table") or para.startswith("Fig")))]

    # Collect embeddings for each paragraph in one place
    para_embeddings = []

    # Embed each paragraph one-by-one
    for para in paras:
        # x = para.find("Keywords")
        # y = para.find("Index Terms")
        
        # if x != -1:
        #     print(para[(x + len("Keywords") + 2):])
        #     return
        # if y != -1:
        #     print(para[(x + len("Index Terms") + 2):])
        #     return
        # else:
        #     continue


        embedding = model.encode([para])
        # reshape (1,n) array to (n,)
        embedding = np.reshape(embedding, (embedding[0].shape))
        para_embeddings.append(embedding)
    
    para_embeddings = np.array(para_embeddings, dtype='float64')
    
    # Mean-pool all the paragraph level embeddings
    # `axis = 0` because for (num_paras, embedding_vec_len) say (73, 384)
    # we want to find the mean of each of the 384 embedding vector values across the 73 paragraphs
    # hence, we will pool vertically to get a final tensor of size (1, embedding_vec_len) i.e. (1, 384)
    pooled_doc_embedding = np.mean(para_embeddings, axis = 0).reshape((1,len(para_embeddings[0])))

    return pooled_doc_embedding


def extract_keywords(TEXT, num_kws, min_kws, max_kws):
    # Set token length of possible keyphrases (via n-gram)
    n_gram_range = (min_kws, max_kws)
    stopwords = "english"

    # Get all possible candidates for keywords and keyphrases (1-n grams)
    count = CountVectorizer(ngram_range = n_gram_range, stop_words = stopwords).fit([TEXT])
    kw_candidates = count.get_feature_names_out()

    # Instantiate SentenceTransformer to derive embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Check whether document is too long
    num_tokens = len(nltk.word_tokenize(TEXT))
    # print(num_tokens, end = "\n\n")

    # Derive embeddings for the document
    if num_tokens > 300:
        doc_embedding = encode_long_document(model, TEXT)
    else: doc_embedding = model.encode([TEXT])
    # Derive embeddings for each candidate keyphrase
    candidate_embeddings = model.encode(kw_candidates)

    # Set num of candidate keyphrases to be shortlisted
    num_candidates = int(250/num_kws)
    # Set num of final keywords required
    top_n = num_kws

    extracted_keywords = max_sum_similarity(doc_embedding, candidate_embeddings, kw_candidates, num_candidates, top_n)
    return extracted_keywords


def recommend_similar_papers(keywords, num_recs=5):
    """
        INPUT:
            keywords: List of Keywords (each a String)
            num_recs: Number of Papers to Recommend (an Integer)
        OUTPUT:
            recommendations: List of similar papers (by Title)
    """
    import json
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Convert list of keywords into a singular string by joining via a separator
    keyword_str = ' ; '.join(keywords)
    
    # Get the embedding for the keywords for this paper
    model = SentenceTransformer("all-MiniLM-L6-v2")
    cur_paper_embeddings = model.encode([keyword_str])
    # Fetch pre-computed embeddings for each of the 363,589 research papers in our Dataset
    cand_paper_embeddings = np.load('/app/easydocs/Assets/cand_paper_embeddings.npy')
    
    # Compute the similarity between this paper and each of the papers in our Dataset
    distances = cosine_similarity(cur_paper_embeddings, cand_paper_embeddings)
    # Fetch the num_recs most similar ones
    with open('/app/easydocs/Assets/training_new.json', 'r') as json_file: json_data = json.load(json_file)
    most_similar_papers = [json_data['data'][index] for index in distances.argsort()[0][-num_recs:]]

    recommendations = [paper['title'] for paper in most_similar_papers]
    return recommendations


# -------------------------------------------------------------------------------------

import os

st.markdown(os.getcwd())
st.markdown(os.listdir())

with st.expander("Keep in mind..."):
    st.markdown("1. For general-purpose texts, use bart-large-cnn.\n2. For academic or scientific texts, use bart-easydocs.\n3. The summary produced may not accurately cover all relevant parts of a text. Use this tool only as a starting guide.\n")

st.subheader("Enter text to extract keywords from")
TEXT = st.text_area(label="dont show", height=150, label_visibility="collapsed")

num_kws_col, min_kws_col, max_kws_col = st.columns(3)

with num_kws_col:
    num_kws = st.slider("Select number of keywords/keyphrases to extract", min_value=1, step=1, max_value=10, key='first', value=1)

with min_kws_col:
    min_kws = st.slider("Select minimum number of words in each keyphrase", min_value=1, step=1, max_value=5, key='second', value=1)

with max_kws_col:
    max_kws = st.slider("Select maximum number of words in each keyphrase", min_value=1, step=1, max_value=5, key='third', value=3)


with st.container():
    btn = st.button("Click here to extract keywords", use_container_width=True, type="primary")
    st.markdown("---")
    
    if btn:
        with st.spinner("Extracting Keywords..."):
            keywords = extract_keywords(TEXT, num_kws, min_kws, max_kws)

        if keywords is not None:
            with st.expander("**Read Keywords**", expanded=True):
                st.markdown(keywords)

            st.markdown("---")
            with st.spinner("Recommending Similar Papers..."):
                papers = recommend_similar_papers(keywords)
                
            if papers is not None:
                with st.expander("**Browse Similar Papers**", expanded=True):
                    st.markdown(papers)
