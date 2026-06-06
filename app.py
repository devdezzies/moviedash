import streamlit as st
from recommender import MovieRecommender


st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide"
)


@st.cache_resource
def load_recommender():
    return MovieRecommender()


recommender = load_recommender()


st.title("🎬 Content-Based Movie Recommender System")
st.write(
    "Aplikasi ini merekomendasikan film berdasarkan kemiripan konten "
    "menggunakan sentence embedding dan cosine similarity."
)

st.divider()

user_id = st.number_input(
    "Masukkan User ID",
    min_value=1,
    step=1
)

top_n = st.slider(
    "Jumlah rekomendasi",
    min_value=5,
    max_value=20,
    value=10
)

if st.button("Generate Recommendation"):
    user_id = int(user_id)

    if not recommender.is_valid_user(user_id):
        st.error("User ID tidak ditemukan di dataset.")

    elif not recommender.is_active_user(user_id):
        st.warning("User ini bukan active user karena jumlah rating kurang dari 20.")

    else:
        summary = recommender.get_user_summary(user_id)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Rated Movies", summary["total_ratings"])

        with col2:
            st.metric("Liked Movies", summary["liked_movies"])

        with col3:
            st.metric("Top Genres", ", ".join(summary["top_genres"][:3]))

        st.subheader("Film yang Pernah Disukai User")
        st.dataframe(
            summary["liked_movie_table"],
            use_container_width=True
        )

        st.subheader("Recommended Movies")

        recommendations = recommender.recommend(
            user_id=user_id,
            top_n=top_n
        )

        if recommendations.empty:
            st.info("Tidak ada rekomendasi yang bisa dibuat untuk user ini.")
        else:
            recommendations = recommendations.copy()
            recommendations.insert(0, "rank", range(1, len(recommendations) + 1))
            recommendations["similarity"] = recommendations["similarity"].round(4)

            st.dataframe(
                recommendations[["rank", "title", "genres", "similarity"]],
                use_container_width=True
            )