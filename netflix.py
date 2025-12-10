import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data/netflix_titles.csv')

print(data.head(10))
print(data.info())

#Type of show
sns.countplot(data=data, x='type', palette='Set2', hue='type')
plt.show()

#Most Producers Countries
data['country'] = data['country'].fillna('Unknown')
top_countries = data['country'].value_counts().head(10)

sns.barplot(x=top_countries.values, y=top_countries.index, palette="magma")
plt.title('Top 10 Countries Producing Netflix Content')
plt.xlabel('Productions Count')
plt.ylabel('Country')
plt.show()

#Realeses Per Year
data['release_year'] = data['release_year'].fillna(0).astype(int)
year_count = data['release_year'].value_counts().sort_index()

sns.lineplot(x=year_count.index, y=year_count.values)
plt.title('Number of shows released per year')
plt.grid(True)
plt.xlabel('Year')
plt.ylabel('Releases')
plt.show()

#Content Ratings
sns.countplot(data=data, x='rating', palette='coolwarm', hue='rating', legend=False)
plt.title('Distribution of Netflix Ratings')
plt.xlabel('Rating')
plt.xticks(rotation=45)
plt.show()

#Movie Duration
movies = data[data['type'] == 'Movie'].copy()
movies['duration'] = movies['duration'].str.replace(' min', '').astype(float)

sns.histplot(x=movies['duration'], bins=30, kde=True, color='green')
plt.title('Netflix Movies Duration')
plt.xlabel('Duration (min)')
plt.show()

#Genre Count
data['genre_split'] = data['listed_in'].str.split(', ')
genre_list = data.explode('genre_split')
top_genres = genre_list['genre_split'].value_counts().head(10)

sns.barplot(x=top_genres.values, y=top_genres.index, palette="viridis")
plt.title('Top 10 Genres on Netflix')
plt.xlabel('Productions Count')
plt.ylabel('Country')
plt.show()
