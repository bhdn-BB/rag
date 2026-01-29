import streamlit as st
import requests
import os
import uuid

# Підтримка Docker environment
API_BASE = os.getenv("API_BASE", "http://localhost:8000")


def format_source(source: dict, idx: int) -> str:
    """Форматує джерело для відображення"""
    parts = [f"**[{idx}] {source['source']}**"]

    if source.get('page') is not None:
        parts.append(f"📄 Сторінка: {source['page']}")

    if source.get('section'):
        parts.append(f"📑 Розділ: {source['section']}")

    if source.get('score') is not None:
        score_percent = source['score'] * 100
        parts.append(f"🎯 Релевантність: {score_percent:.1f}%")

    if source.get('content'):
        parts.append(f"\n> {source['content']}")

    return "\n".join(parts)


def main():
    # -----------------------------
    # Session state ініціалізація
    # -----------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Унікальний ID сесії для backend
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    # -----------------------------
    # Sidebar — Documents
    # -----------------------------
    st.sidebar.title("📚 Управління документами")

    mode = st.sidebar.radio("Додати документ з:", ["URL", "Файлу"])

    if mode == "URL":
        url = st.sidebar.text_input("Введіть URL")
        if st.sidebar.button("➕ Додати URL"):
            try:
                with st.sidebar.spinner("Завантаження..."):
                    r = requests.post(
                        f"{API_BASE}/vector-memory/documents/url",
                        params={"url": url},
                        timeout=60
                    )
                if r.status_code in [200, 202]:
                    data = r.json()
                    st.sidebar.success(f"✅ {data.get('message', 'Додано')}")
                else:
                    st.sidebar.error(f"❌ Помилка: {r.status_code}")
            except Exception as e:
                st.sidebar.error(f"❌ Не вдалося додати: {str(e)}")

    else:
        file = st.sidebar.file_uploader(
            "Завантажити файл",
            type=["pdf", "docx", "txt", "md", "html"],
        )
        if file and st.sidebar.button("➕ Додати файл"):
            try:
                with st.sidebar.spinner("Завантаження..."):
                    r = requests.post(
                        f"{API_BASE}/vector-memory/documents/file",
                        files={"file": file},
                        timeout=120
                    )
                if r.status_code in [200, 202]:
                    st.sidebar.success("✅ Файл додається у фоні")
                else:
                    st.sidebar.error(f"❌ Помилка: {r.status_code}")
            except Exception as e:
                st.sidebar.error(f"❌ Не вдалося завантажити: {str(e)}")

    st.sidebar.divider()

    # -----------------------------
    # Documents status
    # -----------------------------
    if st.sidebar.button("🔄 Оновити статус"):
        st.rerun()

    try:
        status = requests.get(f"{API_BASE}/vector-memory/status", timeout=5).json()
        st.sidebar.markdown("### 📊 Статус бази знань")
        st.sidebar.metric("Фрагментів у базі", status.get('num_documents', 0))

        if status.get('has_cross_encoder'):
            st.sidebar.info("✅ Reranking увімкнено")

    except Exception as e:
        st.sidebar.warning(f"⚠️ База недоступна: {str(e)}")

    # -----------------------------
    # Clear documents
    # -----------------------------
    if st.sidebar.button("🗑 Очистити всі документи", type="secondary"):
        try:
            requests.delete(f"{API_BASE}/vector-memory/clear", timeout=10)
            st.sidebar.success("✅ Базу очищено")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"❌ Помилка: {str(e)}")

    st.sidebar.divider()

    # -----------------------------
    # New conversation - RESET через новий session_id
    # -----------------------------
    if st.sidebar.button("🔁 Нова розмова", type="primary"):
        # Генеруємо НОВИЙ session_id = backend автоматично створить новий агент
        st.session_state.session_id = str(uuid.uuid4())
        # Очищаємо історію UI
        st.session_state.messages = []
        st.sidebar.success("✅ Розпочато нову розмову")
        st.rerun()

    # =============================
    # Main — Chat
    # =============================
    st.title("💬 RAG чат з джерелами")
    st.caption(f"🔗 Підключено до: {API_BASE}")
    st.caption(f"🆔 Сесія: {st.session_state.session_id[:8]}...")

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Показуємо джерела
            if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
                with st.expander(f"📚 Джерела ({len(msg['sources'])})"):
                    for idx, source in enumerate(msg["sources"], 1):
                        st.markdown(format_source(source, idx))
                        if idx < len(msg["sources"]):
                            st.divider()

    # Chat input
    user_input = st.chat_input("Задайте питання на основі документів...")

    if user_input:
        # Show user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        with st.chat_message("user"):
            st.markdown(user_input)

        # Call backend з session_id
        with st.chat_message("assistant"):
            with st.spinner("🤔 Шукаю відповідь..."):
                try:
                    # Відправляємо запит з session_id
                    r = requests.post(
                        f"{API_BASE}/agent/chat",
                        json={
                            "query": user_input,
                            "session_id": st.session_state.session_id  # ← Передаємо сесію
                        },
                        timeout=60
                    )

                    if r.status_code == 200:
                        data = r.json()
                        answer = data.get("answer", "Немає відповіді")
                        sources = data.get("sources", [])
                        rewrite_attempts = data.get("rewrite_attempts", 0)
                        query_rewritten = data.get("query_rewritten")

                        # Показуємо відповідь
                        st.markdown(answer)

                        # Показуємо джерела
                        if sources:
                            with st.expander(f"📚 Джерела ({len(sources)})", expanded=True):
                                for idx, source in enumerate(sources, 1):
                                    st.markdown(format_source(source, idx))
                                    if idx < len(sources):
                                        st.divider()
                        else:
                            # Немає джерел
                            if "немає" in answer.lower():
                                st.info("💡 Спробуйте переформулювати питання або додати більше документів")

                        # Додаткова інформація
                        with st.expander("ℹ️ Детальна інформація", expanded=False):
                            if rewrite_attempts > 0:
                                st.write(f"🔄 Запит було переформульовано {rewrite_attempts} раз(и)")

                            if query_rewritten and query_rewritten != user_input:
                                st.write("🔍 Оптимізований запит:")
                                st.code(query_rewritten)

                            st.write(f"📊 Знайдено джерел: {len(sources)}")

                        # Зберігаємо в історію
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })

                    else:
                        error_msg = f"❌ Помилка: {r.status_code}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })

                except Exception as e:
                    error_msg = f"❌ Не вдалося отримати відповідь: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    st.set_page_config(
        page_title="RAG чат з джерелами",
        layout="wide",
        page_icon="💬",
        initial_sidebar_state="expanded"
    )
    main()