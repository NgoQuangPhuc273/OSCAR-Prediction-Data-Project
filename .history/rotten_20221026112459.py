# # importing the module
import imdb
# creating an instance of the IMDB()
ia = imdb.IMDb()
# Using the Search movie method
items = ia.search_movie('Batman')
for i in items:
	print(i)
