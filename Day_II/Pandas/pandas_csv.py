import pandas as pd
import matplotlib.pylab as plt

PATH = "/home/user/Desktop/NCIT/Day_II/Pandas/"

df = pd.read_csv(PATH + 'weight_height_data.csv')

print (df['weight'])

plt.scatter(df['weight'], df['height'])
plt.show()
