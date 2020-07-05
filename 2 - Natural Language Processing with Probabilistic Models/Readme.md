# Natural Language Processing with Probabilistic Model
Welcome to the [second course](https://www.coursera.org/learn/probabilistic-models-in-nlp) of the Natural Language Processing Specialization at [Coursera](https://www.coursera.org/specializations/natural-language-processing) which is moderated by [DeepLearning.ai](http://deeplearning.ai/). The course is taught by Younes Bensouda Mourri, Łukasz Kaiser and Eddy Shyu.

## Table of contents
- [Natural Language Processing with Probabilistic Model](#natural-language-processing-with-probabilistic-model)
  - [Table of contents](#table-of-contents)
  - [Course summary](#course-summary)
  - [Autocorrect and Dynamic Programming](#autocorrect-and-dynamic-programming)
    - [Autocorrect](#autocorrect)
    - [Building the model](#building-the-model)
    - [Minimum edit distance](#minimum-edit-distance)
    - [Minimum edit distance algorithm](#minimum-edit-distance-algorithm)

## Course summary
This is the  course summary as its given on the course [link] (https://www.coursera.org/learn/probabilistic-models-in-nlp):

In Course 2 of the Natural Language Processing Specialization, offered by deeplearning.ai, you will:

> Create a simple auto-correct algorithm using minimum edit distance and dynamic programming,
> Apply the Viterbi Algorithm for part-of-speech (POS) tagging, which is important for computational linguistics,
> Write a better auto-complete algorithm using an N-gram language model, and 
> Write your own Word2Vec model that uses a neural network to compute word embeddings using a continuous bag-of-words model.
 
> Please make sure that you’re comfortable programming in Python and have a basic knowledge of machine learning, matrix multiplications, and conditional probability.

> By the end of this Specialization, you will have designed NLP applications that perform question-answering and sentiment analysis, created tools to translate languages and summarize text, and even built a chatbot!

> This Specialization is designed and taught by two experts in NLP, machine learning, and deep learning. Younes Bensouda Mourri is an Instructor of AI at Stanford University who also helped build the Deep Learning Specialization. Łukasz Kaiser is a Staff Research Scientist at Google Brain and the co-author of Tensorflow, the Tensor2Tensor and Trax libraries, and the Transformer paper.

## Autocorrect and Dynamic Programming
### Autocorrect
- Autocorrect is an application that changes misspelled words into the correct ones.
  - Example: Happy birthday *deah* friend! ==> dear
- How it works:
  1. Identify a misspelled word
  2. Find strings n edit distance away
  3. Filter candidates
  4. Calculate word probabilities 
### Building the model
- Identify a misspelled word
  - If word not in vocabulary then its misspedlled 
- Find strings n edit distance away
  - Edit: an operation performed on a string to change it
    - how many operations away one string is from another
      - Insert (add a letter)
         - Add a letter to a string at any position: to ==> top,two,...
      -  Delete (remove a letter)
         - Rmove a letter  from a string : hat ==> ha, at, ht
      -  Switch (swap 2 adjacent letters)
         - Exmaple: eta=> eat,tea
      -  Replace (change 1 letter to another)
         - Example: jaw ==> jar,paw,saw,...
 -  By combining the 4 edit operations, we get list of all possible strings that are n edit.
    - ![](Images/1.png)
- Filter candidates: 
  - From the list from step 2, consider only real and correct spelled word
  - if the edit word not in vocabualry ==> remove it from list of candidates
    - ![](Images/2.png)
- Calculate word probabilities: the word candidate is the one with the highest probability 
  - a word probablity in a corpus is: number of times the word appears divided by the total number of words.
    - ![](Images/3.png) 
### Minimum edit distance
- Evaluate the similarity between 2 strings
- Minimum number of edits needed to transform 1 string into another
- the algorithm try to minimize the edit cost  
  - ![](Images/4.png)
- Applications:
  -  Spelling correction
  -  document similarity
  -  machine translation
  -  DNA sequencing
  -  etc
### Minimum edit distance algorithm
- The source word layed on the column
- The target word layed on the row
- Empty string at the start of each word at (0,0)
- D[i,j] is the minimum editing distance between the beginning of the source word to index i and the beginning of the target word to index j 
  - ![](Images/5.png)
- To fillout the rest of the table we can use this formulaic approach:
  - ![](Images/6.png)