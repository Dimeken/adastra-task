import pandas as pd
import json
import logging
from sklearn import preprocessing as p

pd.set_option('display.max_colwidth', None)

logging.basicConfig(filename='movie_dataset.log', level=logging.INFO)


class MovieAnalyzer:
    def __init__(self, movies_file, ratings_file):
        self.movies_file = movies_file
        self.ratings_file = ratings_file
        self.movies = None
        self.ratings = None

    def load_data(self):
        """
        Task 1: Load the dataset from a CSV file into a pandas DataFrame.
        """
        try:
            self.movies = pd.read_csv(self.movies_file)
            self.ratings = pd.read_csv(self.ratings_file)
            
            logging.info("----- Task 1 -----")
            logging.info(f"Data loaded successfully.")
        except FileNotFoundError:
            logging.error(f"File not found.")
        except Exception as e:
            logging.error(f"Error occurred while reading files: {str(e)}")

    def process_data(self):
        """
        Do some transformations on the loaded data and clean them.
        """
        try:
            movies_columns = ['id', 'genres', 'title', 'release_date', 'status']
            self.movies = self.movies[movies_columns].rename(columns={'id': 'movieId'})
            self.movies = self.movies.drop_duplicates()

            self.ratings = self.ratings[['movieId', 'rating']]

            self.ratings_grouped = self.ratings.groupby('movieId')['rating'].mean()
            self.ratings_grouped = self.ratings_grouped.reset_index().rename(columns={'rating': 'avg_rating'})
            self.ratings_grouped['avg_rating'] = round(self.ratings_grouped['avg_rating'], 2)

            self.movies = self.movies.merge(self.ratings_grouped, on='movieId', how='left')

            logging.info(f"Transformations made successfully.")
        except Exception as e:
            logging.error(f"Error occurred while processing data: {str(e)}")

    def print_number_of_movies(self):
        """
        Task 2: Print the number of movies in the dataset.
        """
        try:
            num_movies = self.movies.shape[0]
            logging.info("----- Task 2 -----")
            logging.info(f"Number of movies: {num_movies}")
        except Exception as e:
            logging.error(f"Error occurred while printing number of movies: {str(e)}")

    def print_average_rating(self):
        """
        Task 3: Print the average rating of all the movies.
        Only 7565 movieIds match to each other in movies_metadata.csv and ratings.csv.
        Hence I decided to show two versions of the answer:
        a) avg rating of all movies from ratings.csv
        b) avg rating of all movies from movies_metadata.csv found in ratings.csv
        """
        try:
            logging.info("----- Task 3 -----")
            # 3a:
            logging.info(f"Average rating of all movies from ratings.csv: {self.ratings['rating'].mean():.2f}")
            # 3b:
            logging.info(f"Average rating of all movies from movies_metadata.csv found in ratings.csv: {self.ratings.merge(self.movies, on='movieId')['rating'].mean():.2f}")
        except Exception as e:
            logging.error(f"Error occurred while printing average ratings: {str(e)}")

    def print_top_rated_movies(self):
        """
        Task 4: Print the top 5 highest rated movies

        If we just use average rating, then we miss valuable information about number of ratings.
        Rating value and number of ratings should have equal weight in comparing movies. Hence I
        want to logging.info two results.
        a) Rate movies by average rating
        b) Calculate rate by this algorithm:
            1) Convert 0 to 5 rating to -2.5 to +2.5 rating.
               By this a movie with 1000 1 star rating will not beat a movie with 100 5 stars.
            2) Calculate sum of the balanced rating
            3) Convert it to 0 to 5 scale with sklearn.MinMaxScaler
        """
        try:
            logging.info("----- Task 4 -----")
            # 4a:
            logging.info(f"Top 5 rated movies, version 1:\n{self.movies[['movieId', 'title', 'avg_rating']].sort_values(by='avg_rating', ascending=False).head().to_string(index=False)}")
            # 4b:
            self.movie_ratings = self.ratings.merge(self.movies[['movieId', 'title', 'avg_rating']], on='movieId')
            self.movie_ratings['balanced_rating'] = self.movie_ratings['rating'] - 2.5
            self.movie_ratings = self.movie_ratings.groupby(by=['movieId', 'title', 'avg_rating'])[
                'balanced_rating'].sum().reset_index()

            min_max_scaler = p.MinMaxScaler(feature_range=(0, 5))
            x = self.movie_ratings['balanced_rating'].values.reshape(-1, 1)
            self.movie_ratings['final_rating'] = min_max_scaler.fit_transform(x)

            self.movie_ratings['final_rating'] = round(self.movie_ratings['final_rating'], 2)
            self.movie_ratings['avg_rating'] = round(self.movie_ratings['avg_rating'], 2)

            movie_ratings = self.movie_ratings[['movieId', 'title', 'avg_rating', 'final_rating']].sort_values(
                'final_rating', ascending=False)

            logging.info(f"Top 5 rated movies, version 2:\n{movie_ratings.head().to_string(index=False)}")
        except Exception as e:
            logging.error(f"Error occurred while printing top 5 rated movies: {str(e)}")

    def print_movies_per_year(self):
        """
        Task 5: Print the number of movies released each year
        """
        try:
            pd.set_option('display.max_rows', None)

            self.movies_by_year = self.movies
            self.movies_by_year['release_year'] = pd.DatetimeIndex(self.movies_by_year['release_date']).year.astype('Int64')
            self.movies_by_year = self.movies_by_year.loc[self.movies_by_year['status'] == 'Released']
            self.movies_by_year = self.movies_by_year.groupby('release_year')['movieId'].count()
            self.movies_by_year = self.movies_by_year.reset_index().rename(columns={'movieId': 'movies_count'})

            logging.info("----- Task 5 -----")
            logging.info(f"Movies released each year:\n{self.movies_by_year.to_string(index=False)}")
        except Exception as e:
            logging.error(f"Error occurred while printing movies per year: {str(e)}")

    def print_movies_per_genre(self):
        """
        Task 6: Print the number of movies in each genre
        """
        try:
            pd.set_option('display.max_rows', 20)

            self.genres = self.movies[['movieId', 'genres']]
            self.genres['genres'] = self.genres['genres'].str.replace("'", '"')

            self.genres = pd.json_normalize(self.genres['genres'].apply(json.loads))
            self.genres = pd.concat([self.movies['movieId'], self.genres], axis=1)

            self.genres = pd.melt(self.genres, id_vars='movieId', value_vars=self.genres.columns[1:], var_name='genre_num', value_name='genre')
            self.genres = self.genres.drop(columns=('genre_num'))

            self.genres['genre'] = self.genres['genre'].apply(lambda x: x['name'] if isinstance(x, dict) else None)
            self.genres = self.genres.loc[self.genres['genre'].notna()]
            self.genres = self.genres.groupby('genre').count()

            logging.info("----- Task 6 -----")
            logging.info(f"Movies per genre:\n{self.genres}")
        except Exception as e:
            logging.error(f"Error occurred while printing movies per year: {str(e)}")

    def save_data_to_json(self, output_file):
        """
        Task 7: Save the dataset to a JSON file
        """
        try:
            self.final_df = self.movies.merge(self.movie_ratings[['movieId', 'final_rating']], on='movieId')
            self.final_df.to_json(output_file, orient='records')
            
            logging.info("----- Task 7 -----")
            logging.info("Saved to result.json file successfully.")
        except Exception as e:
            logging.error(f"Error occurred saving result to file: {str(e)}")


def main():
    movies_file =  '..\\data\\movies_metadata_changed.csv'
    ratings_file = '..\\data\\ratings.csv'
    output_file =  '..\\data\\result.json'

    analyzer = MovieAnalyzer(movies_file, ratings_file)
    analyzer.load_data()
    analyzer.process_data()

    analyzer.print_number_of_movies()
    analyzer.print_average_rating()
    analyzer.print_top_rated_movies()
    analyzer.print_movies_per_year()
    analyzer.print_movies_per_genre()

    analyzer.save_data_to_json(output_file)


if __name__ == '__main__':
    main()
