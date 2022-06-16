# Vector Space Semantics to evaluate similarity between Television Series Characters
Created a vector representation of a document containing lines spoken by a character in the Eastenders script data (i.e. from the file `training.csv`).
Then improved that representation such that each character vector is maximially distinguished from the other character documents. 
This distinction is measured by how well a simple information retrieval classification method can select documents from validation and test data as belonging to the correct class of document (i.e. deciding which character spoke the lines by measuring the similarity of those document vectors to those built in training).

As the lines are not evenly distributed in terms of frequency, only used a maximum of the first 400 lines of each character in the training data to create the training documents, and a maximum of the first 40 lines in the test data (from `test.csv`). 
This makes it more challenging, as number of lines spoken by a character can't be used directly or otherwise as a feature.

### Linguistic feature extraction:
Using the first 360 lines from the training split and first 40 lines from the validation split, the pre_process function is modified to include various pre-processing functions such as word tokenizer, punctuation removal, stop words removal, numerals to words conversion, stemming and lemmatization. Lemmatization is more informative than stemming as it looks beyond word reduction and considers vocabulary to apply morphological analysis to words. Therefore, We omitted stemming while calculating the mean rank of 2.625 for the pre-processing step.

Below features for words in character_doc are added in the to_feature_vector_dictionary function.
- Count: Counting the occurrence of words
- POS Tag: part-of-speech tagging for each word
- Previous word :including previous one word along with “PRE_” tag
- Next word : including next one word along with “NEXT_” tag.
- Bigrams

At first we tried adding each key in the feature list with a list of features as the value to it. For example,{‘Well’:[count_1,NEXT_Let], ’Let’:[count_1,PRE_Well]}. This arrangement was not helpful as the values were not numeric and made the matrix more sparse. Instead, {'Well': 1, 'let': 1, 'me': 1, 'know': 1, 'Well@let': 1, 'let@me': 1, 'me@know': 1} incorporating only numeric values such as count of each word and bi grams joined together with @ symbol helped in generating a mean rank of 2.25 ,mean cosine similarity 0.9427518818336169 ,10 characters correctly detected out of 16 and an accuracy: 0.625. We create a pipeline by applying DictVectorizer()and SelectKBest().And experimented with values to obtain optimal value for k=70000( after including tfidf )
     
### Dialogue context data and features:
The line spoken by the characters immediately before and after in terms of the lines spoken by other characters in the same scene can be considered as the context of the line. It can be seen in the train_data data frame that there are different scenes and episodes with scene info. In the character_docs dictionary for each character, the line he/she speaks is the value for that character key. For including the dialogue context, create_character_document_from_dataframe() function is modified. By iterating over each row of the dataframe we try to capture the previous and next lines spoken by other characters. Each identified previously spoken line is then appended to our dictionary value with “" PRE_LINE_" tag and the eol tag is also modified to " _EOL_PRE ".
       
Similarly,Each identified next spoken line is then appended to our dictionary value with " NEXT_LINE_" tag and the eol tag is also modified to " _EOL_NEXT ".By adding this context information, the mean rank increased to 3.25,mean cosine similarity 0.8848562726805826,7 characters detected correctly out of 16 and an accuracy: 0.4375.
 
### Vectorization method:
The aim of using tf-idf instead of the raw frequencies of occurrence of a feature in a document is to lower the impact of tokens that occur very frequently in a given corpus and that are hence empirically less informative than tokens that occur in a small fraction of the training corpus. We make use of TfidfTransformer from sklearn.feature_extraction.text which converts a collection of raw documents to a matrix of TF-IDF features. We apply the TfidfTransformer as one of the transformation in our pipeline. Values for k and mean rank obtained: [k=60,000,mean rank=2.625],[k=66,000,mean rank=2.18],[k=67000,mean rank=1.937] etc. After experimenting different values of K for the k best feature selection, k=70000 is take into consideration, which gives the Mean rank of 1.75,mean cosine similarity 0.4736430892103445,12 characters correctly detected out of 16 and accuracy of 0.75
    
### Testing the best vector representation method:
As for the techniques, For pre-processing made use of word tokenizer, punctuation removal, stop words removal, numerals to words conversion and lemmatization. Secondly,Included count of occurrences of words and bigrams as features. Thereby, made use of context by including previous and next lines in the character document. Lastly, included TfidfTransformer along with SelectKBest in the pipeline. Trained on all of the training data (using the first 400 lines per character maximum) by selecting the best combination of the techniques and finally tested on the test file (using the first 40 lines per character maximum) to obtain mean rank of 1.875, mean cosine similarity = 0.785859552286068, 10 characters correctly detected out of 16 and an accuracy of 0.625.
 




