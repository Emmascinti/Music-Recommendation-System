from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),                     # Pagina principale
    path('find_similar/', views.recommend_songs, name='recommend_songs'),
    path('discover/', views.discover, name='discover'),     # Scopri canzoni nuove
    path('discover/similarity/', views.discover_similarity, name='discover_similarity'),
    path('discover/genre/', views.discover_genre, name='discover_genre'),
    path('mood/', views.mood, name='mood'),
    path('favourites/', views.favourites, name='favourites'),
    path('streaming-trends/', views.streaming_trends, name='streaming_trends'),
    path('mood/morning-coffee/', views.morning_coffee, name='morningCoffee'),
    path('mood/high-energy/', views.high_energy_workout, name='highEnergy'),
    path('mood/chill/', views.chill_and_relax, name='chill'),
    path('mood/feel-good/', views.feel_good_hits, name='feelGood'),
    path('mood/deep-focus/', views.deep_focus, name='deepFocus'),
    path('mood/party/', views.party, name='party'),
    path('mood/late-night-vibes/', views.late_night_vibes, name='lateNightVibes'),
    path('mood/instrumental-chillout/', views.instrumental_chillout, name='instrumentalChillout'),
    path('mood/on-the-road/', views.on_the_road, name='onTheRoad'),
    path('mood/romantic-vibes/', views.romantic_vibes, name='romanticVibes'),
    path('mood/top50/', views.top50, name='top50'),
    path('create_playlist/', views.create_playlist, name='create_playlist'),
]
