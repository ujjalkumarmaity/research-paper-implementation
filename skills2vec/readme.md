## Paper Implementation — Skill2vec: Machine Learning Approach for Determining the Relevant Skills from Job Description

This repository contains the implementation of the Skill2vec model, which is designed to extract similar skills skills of a given skill. The model is based on the Word2Vec architecture and is trained on a dataset of job descriptions to learn the relationships between different skills.

Original paper: [Skill2vec: Machine Learning Approach for Determining the Relevant Skills from Job Description](https://arxiv.org/abs/2006.10724).
blog post: [Paper Implementation — Skill2vec: Machine Learning Approach for Determining the Relevant Skills from Job Description](https://medium.com/@ujjalkumarmaity1998/paper-implementation-skill2vec-machine-learning-approach-for-determining-the-relevant-skills-47a25bbf19d6).

### Requirements
- Python 3.6+
- pandas
- numpy
- torch

### Dataset
Dataset fromat:
```json
[
["Python", "Machine Learning", "Deep Learning", "Data Analysis"],
["Python", "Statistics", "Data Analysis"],
["Java", "Spring", "Hibernate"],
["JavaScript", "React", "Node.js"],
["C++", "Algorithms", "Data Structures"]
]
```

### Training the Model
````bash
python main.py --epochs 30 --batch-size 256 --embedding-dim 300 --window-size 5 --min-count 3 --lr 0.001 --path "data.json"
````
