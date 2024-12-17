import streamlit as st
import tempfile
import os
from gtts import gTTS
from huggingface_hub import hf_hub_download
import joblib

# Multilingual dictionary for translations
translations = {
    "title": {"en": "🎬 Movie Recommendation System", "es": "🎬 Sistema de Recomendación de Películas", "pt": "🎬 Sistema de Recomendação de Filmes"},
    "description": {
        "en": "This system suggests movies based on the chosen genre and a pre-trained model.",
        "es": "Este sistema sugiere películas según el género elegido y un modelo preentrenado.",
        "pt": "Este sistema sugere filmes com base no gênero escolhido e em um modelo pré-treinado.",
    },
    "choose_genre": {"en": "Choose a movie genre:", "es": "Elige un género de película:", "pt": "Escolha um gênero de filme:"},
    "play_audio": {"en": "🔊 Click to listen to the selected option:", "es": "🔊 Haz clic para escuchar la opción seleccionada:", "pt": "🔊 Clique para ouvir a opção selecionada:"},
    "get_recommendations": {"en": "Get Recommendations", "es": "Obtener Recomendaciones", "pt": "Obter Recomendações"},
    "loading_model": {"en": "🔍 Generating movie recommendations...", "es": "🔍 Generando recomendaciones de películas...", "pt": "🔍 Gerando recomendações de filmes..."},
    "recommendation_success": {"en": "✨ Here are your movie recommendations:", "es": "✨ Aquí están tus recomendaciones de películas:", "pt": "✨ Aqui estão suas recomendações de filmes:"},
    "no_recommendations": {"en": "⚠ No recommendations found for the selected genre.", "es": "⚠ No se encontraron recomendaciones para el género seleccionado.", "pt": "⚠ Nenhuma recomendação encontrada para o gênero selecionado."},
    "footer": {"en": "📽️ Movie Recommendation System", "es": "📽️ Sistema de Recomendación de Películas", "pt": "📽️ Sistema de Recomendação de Filmes"},
    "genres": {
        "en": ["Comedy", "Horror", "Science Fiction", "Action", "Thriller", "Mystery", "Documentary"],
        "es": ["Comedia", "Terror", "Ciencia Ficción", "Acción", "Suspense", "Misterio", "Documental"],
        "pt": ["Comédia", "Terror", "Ficção Científica", "Ação", "Thriller", "Mistério", "Documentário"],
    }
}

# Function to play audio
def play_audio(text, lang):
    tts = gTTS(text=text, lang=lang)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        tts.save(temp_audio.name)
        st.audio(temp_audio.name, format="audio/mp3")
        st.session_state["temp_audio_file"] = temp_audio.name  # Save the temporary path

# Function to delete temporary audio files
def cleanup_audio():
    temp_audio_file = st.session_state.get("temp_audio_file")
    if temp_audio_file and os.path.exists(temp_audio_file):
        os.remove(temp_audio_file)

# Function to load the model
@st.cache_resource
def load_model(repo_id, filename, repo_type="model"):
    try:
        file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
        model = joblib.load(file_path)
        st.success(f"✅ Model `{filename}` loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading `{filename}`. Details: {e}")
        return None

# Centralized mapping of genres to Portuguese (keys in 'movies')
genre_mapping = {
    "en": {
        "Comedy": "Comédia", 
        "Horror": "Terror", 
        "Science Fiction": "Ficção Científica", 
        "Action": "Ação", 
        "Thriller": "Thriller", 
        "Mystery": "Mistério", 
        "Documentary": "Documentário"
    },
    "es": {
        "Comedia": "Comédia", 
        "Terror": "Terror", 
        "Ciencia Ficción": "Ficção Científica", 
        "Acción": "Ação", 
        "Suspense": "Thriller", 
        "Misterio": "Mistério", 
        "Documental": "Documentário"
    },
    "pt": {
        "Comédia": "Comédia", 
        "Terror": "Terror", 
        "Ficção Científica": "Ficção Científica", 
        "Ação": "Ação", 
        "Thriller": "Thriller", 
        "Mistério": "Mistério", 
        "Documentário": "Documentário"
    }
}

# Updated function to simulate movie recommendations
def generate_recommendations(genre, vectorizer, similarity, lang_key):
    try:
        st.info(translations["loading_model"][lang_key])
        
        # Simulated movie data (Portuguese keys only)
        movies = {
            "Comédia": ["Superbad", "The Hangover", "Jumanji"],
            "Terror": ["The Conjuring", "Insidious", "Get Out"],
            "Ficção Científica": ["Interstellar", "Inception", "Blade Runner"],
            "Ação": ["Fast & Furious", "John Wick", "Mad Max"],
            "Thriller": ["Gone Girl", "Prisoners", "The Girl with the Dragon Tattoo"],
            "Mistério": ["Knives Out", "Sherlock Holmes", "The Prestige"],
            "Documentário": ["The Social Dilemma", "Planet Earth", "13th"]
        }

        # Map the translated genre to Portuguese
        genre_in_pt = genre_mapping[lang_key].get(genre, "")
        
        # Get recommendations
        recommendations = movies.get(genre_in_pt, [])
        
        if recommendations:
            st.success(translations["recommendation_success"][lang_key])
            for movie in recommendations:
                st.write(f"📽️ {movie}")
        else:
            st.warning(translations["no_recommendations"][lang_key])
    except Exception as e:
        st.error(f"❌ An error occurred while generating recommendations: {e}")

# Streamlit Main Interface
# Language selection
language = st.sidebar.selectbox("🌐 Choose your language:", ["English", "Español", "Português"])
lang_code = {"English": "en", "Español": "es", "Português": "pt"}[language]
current_lang = lang_code

# Load models
repo_id_vectorizer = "RitaViegas/vectorizer.pkl"
repo_id_similarity = "RitaViegas/similarity.pkl"
vectorizer = load_model(repo_id_vectorizer, "vectorizer.pkl")
similarity = load_model(repo_id_similarity, "similarity.pkl")

# Verify models
if vectorizer is None or similarity is None:
    st.stop()

st.title(translations["title"][current_lang])
st.markdown(translations["description"][current_lang])

# Movie genre options
# Movie genre options
genres = translations["genres"][current_lang]  # Gêneros traduzidos
selected_genre = st.selectbox(translations["choose_genre"][current_lang], genres)

# Play audio of the selected genre
st.write(translations["play_audio"][current_lang])
play_audio(f"Você escolheu o gênero {selected_genre}", lang_code)

# Generate recommendations
if st.button(translations["get_recommendations"][current_lang]):
    generate_recommendations(selected_genre, vectorizer, similarity, current_lang)

cleanup_audio()

st.markdown("---")
st.caption(translations["footer"][current_lang])








