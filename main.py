import streamlit as st
from recommendation_engine import load_resources, get_recommendations, get_bn_metadata

def main():
    st.title("📚 System Rekomendacji Książek")

    try:
        book_data, tfidf, indices, tfidf_matrix = load_resources()
        st.success("✅ Zasoby załadowane pomyślnie!")
        st.write(f"Liczba książek w bazie: {len(book_data)}")
        st.write(f"Rozmiar macierzy TF-IDF: {tfidf_matrix.shape}")

        # Wybór tytułu z listy (bez literówek)
        selected_title = st.selectbox(
            "Wybierz książkę:",
            options=book_data['Title'].str.title().sort_values().unique(),
            index=0
        )

        if st.button("Generuj rekomendacje"):
            st.write(f"**Wybrany tytuł:** {selected_title}")

            try:
                recommendations = get_recommendations(
                    selected_title.lower(),
                    book_data,
                    indices,
                    tfidf_matrix
                )

                if not recommendations.empty:
                    st.success("🎯 Znaleziono rekomendacje:")
                    for _, row in recommendations.iterrows():
                        with st.expander(f"{row['Title'].title()} - {row['Author']}"):
                            st.write(f"**Kategoria:** {row['Category']}")
                            st.write(f"**Opis:** {row['Description']}")
                            metadata = get_bn_metadata(row['ISBN'])
                            if metadata:
                                cover_url = metadata.get('cover', '')
                                if cover_url:
                                    st.image(cover_url, width=200)
                                st.write(f"**Wydawca:** {metadata.get('publisher', 'brak danych')}")
                                st.write(f"**Rok wydania:** {metadata.get('publicationYear', 'brak danych')}")
                else:
                    st.warning("⚠️ Brak rekomendacji dla wybranego tytułu.")

            except Exception as e:
                st.error(f"❌ Krytyczny błąd: {str(e)}")

    except Exception as e:
        st.error(f"❌ Błąd inicjalizacji: {str(e)}")
        st.write("**Możliwe przyczyny:**")
        st.markdown("""
            - Plik CSV nie istnieje lub ma złą strukturę
            - Brak wymaganych kolumn w pliku CSV
            - Błąd kompilacji modelu TF-IDF
        """)

if __name__ == "__main__":
    main()
