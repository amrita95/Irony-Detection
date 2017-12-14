import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = {'Classifiers':['Decision Tree','Random Forest','GaussianNB','Linear SVM'],
        'Accuracy':[ 56.4,59.8,57.1,63.92],
         'Recall':[ 55.8,57.1,73.2,61.85],
        'Precision':[56.35,60.2,55.3,64.37],
        'F-score':[ 56.12,58.66,62.9,63.09]}
df = pd.DataFrame(data, columns = ['Classifiers', 'Accuracy', 'Recall', 'Precision','F-score'])
print(df)

pos = list(range(len(df['Accuracy'])))
width = 0.20
fig, ax = plt.subplots(figsize=(10,5))

plt.bar(pos,
        df['Accuracy'],
        width,
        alpha=0.5,
        color='#EE3224',
        label=df['Classifiers'][0])

plt.bar([p + width for p in pos],
        df['Recall'],
        width,
        alpha=0.5,
        color='#2C3221',
        label=df['Classifiers'][1])
plt.bar([p + width*2  for p in pos],
        df['Precision'],
        width,
        alpha=0.5,
        color='#F78F1E',
        label=df['Classifiers'][2])
plt.bar([p + width*3 for p in pos],
        df['Accuracy'],
        width,
        alpha=0.5,
        color='#FFC222',
        label=df['Classifiers'][3])

ax.set_ylabel('Scores')

# Set the chart's title
ax.set_title('Model Comparison')

# Set the position of the x ticks
ax.set_xticks([p + 1.5 * width for p in pos])

# Set the labels for the x ticks
ax.set_xticklabels(df['Classifiers'])

# Setting the x-axis and y-axis limits
plt.xlim(min(pos)-width, max(pos)+width*4)
plt.ylim([0, 100])

# Adding the legend and showing the plot
plt.legend(['Accuracy', 'Recall', 'Precision', 'F-score'], loc='upper left')
plt.grid()
plt.show()
