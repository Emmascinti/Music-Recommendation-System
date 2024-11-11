# Music Recommendation System

This project is a graph-based music recommendation system that leverages Neo4j to analyze and recommend songs based on user preferences and song features. 
The Django-based application supports various recommendation types, including content-based recommendations based on musical features.
![image](https://github.com/user-attachments/assets/b2da5375-f185-49af-9926-675f57d7dfa3)

## Features

- Recommend similar songs based on cosine similarity between features like danceability, energy, valence, and tempo.
- Generate custom playlists.
- Explore recommendations using various filters and criteria.

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- Django
- Neo4j
- Neo4j Graph Data Science (GDS) Library

### Step-by-Step Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd music_recommender_project
   ```
2. **Install dependencies**

Make sure to have a virtual environment activated, then install required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Neo4j Database**
   
- Install Neo4j and set up a local database instance.
- Add the dataset to your Neo4j instance, the instruction are in the graph_modeling_and_analysis.ipynb file.
- Enable the Neo4j Graph Data Science (GDS) library.

5. **Configure Neo4j Credentials**
   
In the Django settings file (settings.py), add your Neo4j credentials:

   ```bash
  NEO4J_URI = "bolt://localhost:7687"  # or your Neo4j instance URL
  NEO4J_USER = "<your-neo4j-username>"
  NEO4J_PASSWORD = "<your-neo4j-password>"
   ```

5. **Run Database Migrations**

   ```bash
   python manage.py migrate
   ```
   
7. **Start the Development Server**
   
   ```bash
   python manage.py runserver
   ```
   
9. **Access the Application**
Open your web browser and go to http://127.0.0.1:8000 to access the music recommendation platform.
   


