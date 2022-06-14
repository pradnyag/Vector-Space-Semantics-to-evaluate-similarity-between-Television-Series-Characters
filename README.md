# Vector Space Semantics to evaluate similarity between Television Series Characters
Created a vector representation of a document containing lines spoken by a character in the Eastenders script data (i.e. from the file `training.csv`).
Then improved that representation such that each character vector is maximially distinguished from the other character documents. 
This distinction is measured by how well a simple information retrieval classification method can select documents from validation and test data as belonging to the correct class of document (i.e. deciding which character spoke the lines by measuring the similarity of those document vectors to those built in training).
As the lines are not evenly distributed in terms of frequency, only used a maximum of the first 400 lines of each character in the training data to create the training documents, and a maximum of the first 40 lines in the test data (from `test.csv`). 
This makes it more challenging, as number of lines spoken by a character can't be used directly or otherwise as a feature.


