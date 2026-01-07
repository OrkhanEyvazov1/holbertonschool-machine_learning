#!/usr/bin/env python3

'''bar chart example'''
import numpy as np
import matplotlib.pyplot as plt


def bars():
    '''Bar chart example'''
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))
    people = ['Farrah', 'Fred', 'Felicia']
    fruit_types = ['apples', 'bananas', 'oranges', 'peaches']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
    x = np.arange(len(people))
    bottom = np.zeros(len(people))
    for counts, label, color in zip(fruit, fruit_types, colors):
        plt.bar(x, counts, bottom=bottom, width=0.5, color=color, label=label)
        bottom += counts
    plt.xticks(x, people)
    plt.yticks(np.arange(0, 81, 10))
    plt.ylim(0, 80)
    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.legend()
    plt.tight_layout()
    plt.show()
