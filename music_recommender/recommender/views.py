from django.shortcuts import render
from neo4j import GraphDatabase
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import datetime


class ContentBasedRecommender:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def get_similar_songs(self, song_title):
        # Returns 50 songs similar to the specified one based on cosine similarity
        query = '''
        MATCH (s1:Song {title: $song_title})
        MATCH (s2:Song)-[:PERFORMED_BY]->(p:Performer)
        WHERE s1 <> s2
        WITH s1, s2, p,
            [s1.danceability, s1.energy, s1.valence, s1.tempo] AS vector1, 
            [s2.danceability, s2.energy, s2.valence, s2.tempo] AS vector2
        WITH s1, s2, p, gds.similarity.cosine(vector1, vector2) AS similarity
        WHERE similarity > 0.7 AND similarity < 1
        RETURN s2.title AS RecommendedSong, p.name AS Artist, s2.preview_url as Url
        ORDER BY similarity DESC
        LIMIT 50;
        '''
        with self.driver.session() as session:
            result = session.run(query, song_title=song_title)
            return [{"RecommendedSong": record["RecommendedSong"], "Artist": record["Artist"], "Url": record["Url"]} for record in result]

    
    def get_recently_listened_songs(self):
        query = '''
        MATCH (s:Song)-[:HAS_STREAMING_EVENT]->(e:StreamingEvent)
        WITH s ORDER BY e.timestamp_central DESC
        RETURN DISTINCT s.title AS title
        LIMIT 20
        '''
        with self.driver.session() as session:
            result = session.run(query)
            return [record['title'] for record in result]
        

    def get_similar_unheard_songs(self, song_title):
        query = '''
        MATCH (s1:Song {title: $song_title})
        MATCH (s2:Song)-[:PERFORMED_BY]->(p:Performer)
        WHERE s1 <> s2
        AND NOT (s2)-[:HAS_STREAMING_EVENT]->()
        WITH s1, s2, p,
            [s1.danceability, s1.energy, s1.valence, s1.tempo] AS vector1,
            [s2.danceability, s2.energy, s2.valence, s2.tempo] AS vector2
        WITH s1, s2, p, gds.similarity.cosine(vector1, vector2) AS similarity
        WHERE similarity > 0.9
        RETURN s2.title AS RecommendedSong, p.name AS Artist, s2.preview_url as Url
        ORDER BY similarity DESC
        LIMIT 2;
        '''
        with self.driver.session() as session:
            result = session.run(query, song_title=song_title)
            return [{"RecommendedSong": record['RecommendedSong'], "Artist": record["Artist"], "Url": record['Url']} for record in result]


    def get_shortest_path_unheard_songs(self, song_title):
        query = '''
        MATCH (s1:Song {title: $song_title}), (s2:Song)
        WHERE NOT (s2)-[:HAS_STREAMING_EVENT]->() AND s1 <> s2
        MATCH path = allShortestPaths((s1)-[*1..2]-(s2)), (s2:Song)-[:PERFORMED_BY]->(p:Performer)
        WITH DISTINCT s2.title AS RecommendedSong, p, s2.preview_url as Url
        RETURN DISTINCT RecommendedSong, p.name AS Artist, Url
        LIMIT 2;
        '''
        with self.driver.session() as session:
            result = session.run(query, song_title=song_title)
            return [{"RecommendedSong": record['RecommendedSong'], "Artist": record["Artist"], "Url": record['Url']} for record in result]

    def get_playlist_by_genre(self):
        # Return last 20 listened songs with genre
        query = '''
        MATCH (s:Song)-[:HAS_STREAMING_EVENT]->(e:StreamingEvent)
        WITH s ORDER BY e.timestamp_central DESC
        MATCH (s)-[:HAS_GENRE]->(g:Genre), (s)-[:PERFORMED_BY]->(p:Performer)
        RETURN DISTINCT s.title AS title, g.name AS genre, p.name AS Artist
        LIMIT 20
        '''
        with self.driver.session() as session:
            result = session.run(query)
            return [{"title": record['title'], "genre": record['genre'], "Artist": record['Artist']} for record in result]


    
    def get_unheard_songs_by_genre(self, genre):
        query = '''                                                                                                             
        MATCH (g:Genre {name: $genre})<-[:HAS_GENRE]-(s:Song)-[:PERFORMED_BY]->(p:Performer)
        WHERE NOT (s)-[:HAS_STREAMING_EVENT]->()
        RETURN s.title AS RecommendedSong, p.name AS Artist, s.preview_url as Url
        LIMIT 10
        '''
        with self.driver.session() as session:
            result = session.run(query, genre=genre)
            return [{"RecommendedSong": record['RecommendedSong'], "Artist": record["Artist"], "Url": record['Url']} for record in result]


    def discover_songs_by_similarity(self):
        recommended_songs_set = set()  # Use a set to avoid duplicates
        recently_listened = self.get_recently_listened_songs()

        for song_title in recently_listened:
            similar_songs = self.get_similar_unheard_songs(song_title)
            
            # Add similar songs to the set
            for song in similar_songs:
                recommended_songs_set.add((song["RecommendedSong"], song["Artist"], song["Url"]))

            # If there are less than 2 similar songs, looks for shortest paths
            if len(similar_songs) < 2:
                shortest_path_songs = self.get_shortest_path_unheard_songs(song_title)
                for song in shortest_path_songs:
                    recommended_songs_set.add((song["RecommendedSong"], song["Artist"], song["Url"])) 

            if len(recommended_songs_set) >= 40:
                break

        if len(recommended_songs_set) < 40:
            for song_title in recently_listened:
                similar_songs = self.get_similar_songs(song_title) 
                for song in similar_songs:
                    recommended_songs_set.add((song["RecommendedSong"], song["Artist"], song['Url']))

                if len(recommended_songs_set) >= 40:
                    break

        # Convert set to list and limit it to 40
        return [{"RecommendedSong": song[0], "Artist": song[1], "Url": song[2]} for song in list(recommended_songs_set)[:40]]

    def discover_songs_by_genre(self):
        recently_listened = self.get_playlist_by_genre()
        
        recommended_songs_set = set()  

        genres = {song['genre'] for song in recently_listened}

        for genre in genres:
            unheard_songs = self.get_unheard_songs_by_genre(genre)
            for song in unheard_songs:
                recommended_songs_set.add((song["RecommendedSong"], song["Artist"], song["Url"]))  

            if len(recommended_songs_set) >= 40:
                break
        return [{"RecommendedSong": song[0], "Artist": song[1], "Url": song[2]} for song in list(recommended_songs_set)[:40]]
    
    def get_streaming_trends(self):
         # Return number of streamings per day
        query = '''
        MATCH (e:StreamingEvent)
        WITH e, apoc.date.parse(e.timestamp_utc, 'ms', 'MM/dd/yyyy hh:mm:ss a') AS timestamp_ms
        WITH datetime({epochMillis: timestamp_ms}) AS timestamp_utc, e
        WITH date(timestamp_utc) AS day
        RETURN day , COUNT(*) AS value
        ORDER BY day;
        '''
        with self.driver.session() as session:
            result = session.run(query)
            data = [{"Date": record['day'], "Count": record["value"]} for record in result]
            return data
    
    def get_top50(self):
        query = '''
        MATCH (s:Song)-[:PERFORMED_BY]->(p:Performer)
        WHERE s.popularity IS NOT NULL
        RETURN s.title AS song_title, p.name as Artist, s.popularity as popularity
        ORDER BY popularity DESC
        LIMIT 50;
        '''
        with self.driver.session() as session:
            result = session.run(query)
            data = [{"title": record['song_title'], "artist": record["Artist"]} for record in result]
            return data
        
    def get_favs(self):
        query = '''
        MATCH (s:Song)-[:HAS_STREAMING_EVENT]->(e:StreamingEvent), (s)-[:PERFORMED_BY]->(p:Performer)
        WITH s, apoc.date.parse(e.timestamp_utc, 'ms', 'MM/dd/yyyy hh:mm:ss a') AS timestamp_ms, p, e
        WITH datetime({epochMillis: timestamp_ms}) AS timestamp_utc, e,p,s
        WITH date(timestamp_utc) AS day,e,p,s
        WHERE day >= date() - duration('P5Y')
        RETURN s.title AS song_title, p.name AS Artist, COUNT(e) AS play_count, s.preview_url as Url
        ORDER BY play_count DESC
        LIMIT 50;
        '''
        with self.driver.session() as session:
            result = session.run(query)
            data = [{"Song": record['song_title'], "Artist": record["Artist"], "Url": record['Url']} for record in result]
            return data


def playlist_by_mood(request, title, description, filters, time_of_day=None, num=50):
    # Playlist generator
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
    
    current_month = datetime.datetime.now().month

    filtered_query = '''
    MATCH (s:Song)-[:PERFORMED_BY]->(p:Performer)
    WHERE ''' + ' AND '.join(filters)
    
    if current_month != 12:
        filtered_query += '''
        AND NOT (s.title CONTAINS 'Christmas' OR
                s.title CONTAINS 'Santa' OR
                s.title CONTAINS 'Noel' OR
                s.title CONTAINS 'Mistletoe' OR
                s.title CONTAINS 'Silent Night' OR
                s.title CONTAINS 'Bell')'''
        
    filtered_query += '''
    RETURN s.title AS title, p.name AS artist, s.popularity AS popularity, s.preview_url as Url
    ORDER BY s.popularity DESC
    LIMIT $num
    '''

    similarity_query = '''
    MATCH (popular:Song)-[:HAS_STREAMING_EVENT]->(e:StreamingEvent)
    WITH apoc.date.parse(e.timestamp_utc, 'ms', 'MM/dd/yyyy hh:mm:ss a') AS timestamp_ms, popular, e
    WITH datetime({epochMillis: timestamp_ms}) AS timestamp_utc, popular, e
    WHERE '''
    
    if time_of_day == 'morning':
        similarity_query += "time(timestamp_utc) >= time('06:00:00') AND time(timestamp_utc) < time('12:00:00')"
    elif time_of_day == 'night':
        similarity_query += "time(timestamp_utc) >= time('22:00:00') OR time(timestamp_utc) < time('04:00:00')"
    
    similarity_query += '''
    WITH popular, COUNT(e) AS listen_count
    ORDER BY listen_count DESC
    LIMIT 5 

    MATCH (s2:Song)-[:PERFORMED_BY]->(p:Performer)
    WHERE NOT (s2)-[:HAS_STREAMING_EVENT]->() AND popular <> s2
    WITH popular, s2, p,
         [popular.danceability, popular.energy, popular.liveness, popular.tempo, popular.acousticness] AS vector1, 
         [s2.danceability, s2.energy, s2.liveness, s2.tempo, s2.acousticness] AS vector2
    WITH popular, s2, p, gds.similarity.cosine(vector1, vector2) AS similarity
    WHERE similarity > 0.75'''
    
    if current_month != 12:
        similarity_query += '''
        AND NOT (s2.title CONTAINS 'Christmas' OR
                 s2.title CONTAINS 'Santa' OR
                 s2.title CONTAINS 'Noel' OR
                 s2.title CONTAINS 'Silent Night' OR
                 s2.title CONTAINS 'Silver Bells')'''
    
    similarity_query += '''    
    RETURN s2.title AS title, p.name AS artist, similarity, s2.preview_url as Url
    ORDER BY similarity DESC
    LIMIT 25
    '''
    
    with driver.session() as session:
       
        if time_of_day != None:
            filtered_results = session.run(filtered_query, num=num)
            filtered_recommendations = [{"title": record["title"], "artist": record["artist"], "Url": record["Url"]} for record in filtered_results]
            
            similarity_results = session.run(similarity_query)
            similarity_recommendations = [{"title": record["title"], "artist": record["artist"], "Url": record["Url"]} for record in similarity_results]
            
            combined_recommendations = filtered_recommendations + similarity_recommendations
        else:
            filtered_results = session.run(filtered_query, num=num)
            filtered_recommendations = [{"title": record["title"], "artist": record["artist"], "Url": record["Url"]} for record in filtered_results]
            combined_recommendations = filtered_recommendations

    driver.close()  
    
    return render(request, 'recommender/playlist.html', {
        'title': title,
        'description': description,
        'recommendations': combined_recommendations
    })


def home(request):
    return render(request, 'recommender/home.html')

def recommend_songs(request):
    recommendations = None
    if request.method == 'POST':
        song_title = request.POST.get('song_title')
        recommender = ContentBasedRecommender("bolt://localhost:7687", "neo4j", "password")
        recommendations = recommender.get_similar_songs(song_title)
        recommender.close()
    return render(request, 'recommender/recommend_songs.html', {'recommendations': recommendations})

def discover(request):
    return render(request, 'recommender/discover.html')

def discover_similarity(request):
    recommender = ContentBasedRecommender("bolt://localhost:7687", "neo4j", "password")
    recommendations = recommender.discover_songs_by_similarity()
    recommender.close()
    return render(request, 'recommender/discover_similarity.html', {'recommendations': recommendations})

def discover_genre(request):
    recommender = ContentBasedRecommender("bolt://localhost:7687", "neo4j", "password")
    recommendations = recommender.discover_songs_by_genre()
    recommender.close()
    return render(request, 'recommender/discover_genre.html', {'recommendations': recommendations})


def favourites(request):
    recommender = ContentBasedRecommender("bolt://localhost:7687", "neo4j", "password")
    recommendations = recommender.get_favs()
    recommender.close()
    return render(request, 'recommender/favourites.html', {'recommendations': recommendations})


def top50(request):
    recommender = ContentBasedRecommender("bolt://localhost:7687", "neo4j", "password")
    recommendations = recommender.get_top50()
    recommender.close()
    title = 'Top 50'
    description = '50 Most popular songs of all time'
    return render(request, 'recommender/playlist.html', {'title': title, 'description':description, 'recommendations': recommendations})


def mood(request):
    return render(request, 'recommender/mood.html')


def morning_coffee(request):
    filters = [
        "s.energy < 0.5",
        "s.acousticness > 0.7",
        "s.tempo < 100"
    ]
    description = "Perfect for starting your day with calm, uplifting tunes. Enjoy a relaxed vibe as you sip your morning coffee."
    return playlist_by_mood(request, "Morning Coffee", description, filters, time_of_day='morning', num = 25)

def late_night_vibes(request):
    filters = [
        "s.energy < 0.5",
        "s.liveness < 0.3",
        "s.tempo < 80"
    ]
    description = "Relax with songs similar to those most enjoyed in the late night."
    return playlist_by_mood(request, "Late Night Vibes", description, filters, time_of_day='night', num = 25)


def high_energy_workout(request):
    filters = [
        "s.energy > 0.8",
        "s.tempo > 120",
        "s.danceability > 0.7",
        "s.loudness > -6 "
    ]
    description = "Perfect for keeping up your motivation during a workout or run, with fast beats and energizing vibes."
    return playlist_by_mood(request, "High-Energy Workout", description, filters)

def chill_and_relax(request):
    filters = [
        "s.energy < 0.4",
        "s.acousticness > 0.6",
        "s.speechiness < 0.3",
        "s.tempo < 100",
        "s.loudness < -10 "

    ]
    description = "A relaxing playlist for winding down, meditating, or studying."
    return playlist_by_mood(request, "Chill & Relax", description, filters)

def feel_good_hits(request):
    filters = [
        "s.valence > 0.7",
        "s.energy >= 0.4 AND s.energy <= 0.7",
        "s.popularity > 50",
        "s.danceability > 0.5",

    ]
    description = "Songs that make you feel happy, upbeat, and ready to tackle the day."
    return playlist_by_mood(request, "Feel-Good Hits", description, filters)

def deep_focus(request):
    filters = [
        "s.instrumentalness > 0.7",
        "s.energy < 0.5",
        "s.speechiness < 0.3",
        "s.loudness < -8 ",
        "s.acousticness > 0.4",
    ]
    description = "Great for studying or working with minimal distractions."
    return playlist_by_mood(request, "Deep Focus", description, filters)

def party(request):
    filters = [
        "s.energy > 0.8",
        "s.danceability > 0.7",
        "s.popularity > 60",
        "s.loudness > -5 "
    ]
    description = "Songs that bring energy and excitement to any party."
    return playlist_by_mood(request, "Party", description, filters)


def instrumental_chillout(request):
    filters = [
        "s.instrumentalness > 0.7",
        "s.acousticness > 0.5",
        "s.speechiness < 0.3"
    ]
    description = "A playlist full of instrumental tracks to create a calm, unobtrusive background."
    return playlist_by_mood(request, "Instrumental Chillout",description,  filters)

def on_the_road(request):
    filters = [
        "s.energy > 0.5",
        "s.valence > 0.5",
        "s.tempo >= 90 AND s.tempo <= 130",
        "s.popularity > 40"
    ]
    description = "Perfect for a road trip, with uplifting and energetic tracks that make long drives enjoyable."
    return playlist_by_mood(request, "On The Road", description, filters)

def romantic_vibes(request):
    filters = [
        "s.valence > 0.5",
        "s.tempo < 90",
        "s.acousticness > 0.4",
        "s.explicit = 'False'"
    ]
    description = "Sweet, romantic songs for a cozy evening."
    return playlist_by_mood(request, "Romantic Vibes", description, filters)

class PlaylistRecommender:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_custom_recommendations(self, danceability, energy, valence, tempo, acousticness):
        query = '''
        MATCH (s1:Song)
        WHERE (s1.danceability >= $danceability - 0.2 AND s1.danceability <= $danceability + 0.2) AND
            (s1.energy >= $energy - 0.2 AND s1.energy <= $energy + 0.2) AND
            (s1.valence >= $valence - 0.2 AND s1.valence <= $valence + 0.2) AND
            (s1.tempo >= $tempo - 10 AND s1.tempo <= $tempo + 10) AND
            (s1.acousticness >= $acousticness - 0.2 AND s1.acousticness <= $acousticness + 0.2) 
        WITH s1, [s1.danceability, s1.energy, s1.valence, s1.tempo] AS vector1

        MATCH (s2:Song)-[:PERFORMED_BY]->(p2:Performer)
        WHERE s1 <> s2
        WITH s1, s2, p2,
            vector1,
            [s2.danceability, s2.energy, s2.valence, s2.tempo] AS vector2
        WITH s1, s2, p2, vector1, vector2,
            gds.similarity.cosine(vector1, vector2) AS similarity
        WHERE similarity > 0.7 AND similarity < 1
        WITH DISTINCT s2.title AS RecommendedSong, p2, s2, similarity
        RETURN RecommendedSong, p2.name AS Artist, s2.preview_url AS Url, similarity
        ORDER BY similarity DESC
        LIMIT 50;
        '''
        params = {
            "danceability": danceability,
            "energy": energy,
            "valence": valence,
            "tempo": tempo,
            "acousticness": acousticness,
        }
        
        with self.driver.session() as session:
            result = session.run(query, params)
            return [{"RecommendedSong": record["RecommendedSong"], "Artist": record["Artist"], "Url": record["Url"]} for record in result]
        
def create_playlist(request):
    recommendations = None
    if request.method == 'POST':
        danceability = float(request.POST.get('danceability', 0.5))
        energy = float(request.POST.get('energy', 0.5))
        valence = float(request.POST.get('valence', 0.5))
        tempo = float(request.POST.get('tempo', 120))
        acousticness = float(request.POST.get('acousticness', 0.5))

        recommender = PlaylistRecommender("bolt://localhost:7687", "neo4j", "password")
        recommendations = recommender.get_custom_recommendations(danceability, energy, valence, tempo, acousticness)
        print(recommendations)
        recommender.close()

    return render(request, 'recommender/create_playlist.html', {'recommendations': recommendations})

def streaming_trends(request):
    # Get year parameter (default to 2021 if not provided)
    year = request.GET.get('year', 2017)

    # Ensure that the year is valid
    if int(year) not in range(2017, 2022):
        year = 2017

    recommender = ContentBasedRecommender("bolt://localhost:7687", "neo4j", "password")
    recommendations = recommender.get_streaming_trends()

    # Convert Neo4j date objects to Python date objects
    data = [(record['Date'].to_native(), record['Count']) for record in recommendations]

    recommender.close()

    # Convert data into a pandas DataFrame
    df = pd.DataFrame(data, columns=["Date", "Count"])

    # Filter the data for the selected year
    df['Date'] = pd.to_datetime(df['Date'])  # Ensure the Date column is datetime-compatible
    df = df[df['Date'].dt.year == int(year)]

    # Create 'Month' and 'Day' columns for plotting purposes
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day

    # Pivot the data to create a matrix where each row is a month and each column is a day
    pivot_table = df.pivot_table(index='Month', columns='Day', values='Count', aggfunc='sum', fill_value=0)
    cmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=False, as_cmap=True)
    # Create a heatmap
    plt.figure(figsize=(12, 4.5))
    sns.heatmap(pivot_table, cmap=cmap, cbar=True, cbar_kws={
    'label': 'Streaming Count',        # Label for the colorbar (acting as a legend)
    'orientation': 'vertical',         # Colorbar is vertical (default)
    'shrink': 0.8,                     # Shrink the colorbar to 80% of its size
    'ticks': [0, 50, 100, 150],        # Custom tick values on the colorbar (adjust as needed)
    'fraction': 0.02,                  # Fraction of the space to allocate to the colorbar
    'pad': 0.05                        # Padding between the heatmap and colorbar
    }, square=True)

    # Save the plot to a BytesIO object and encode it to base64 for embedding in HTML
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plot_url = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    img_buffer.close()

    # Pass data and the plot URL to the template
    return render(request, 'recommender/streaming_trends.html', {
        'plot_url': plot_url,
        'year': year,
    })