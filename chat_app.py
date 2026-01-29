import streamlit as st
import requests
import os
import uuid

# -----------------------------
# Підтримка Docker environment
# -----------------------------
API_BASE = os.getenv("API_BASE", "http://localhost:8000")


def format_source(source: dict, idx: int) -> str:
    """Форматує джерело для відображення з урахуванням типу"""
    source_type = source.get('source_type', 'document')

    if source_type == "url":
        # Для URL показуємо тільки посилання
        parts = [f"**[{idx}] 🔗 URL**"]
        parts.append(f"[{source['source']}]({source['source']})")

        if source.get('score') is not None:
            score_percent = source['score'] * 100
            parts.append(f"🎯 Релевантність: {score_percent:.1f}%")

        if source.get('content'):
            parts.append(f"\n> {source['content'][:300]}...")

    else:
        # Для документів показуємо повну інформацію
        parts = [f"**[{idx}] 📄 {source['source']}**"]

        doc_info = []
        if source.get('page') is not None:
            doc_info.append(f"Сторінка {source['page']}")

        if source.get('section'):
            doc_info.append(f"Розділ: {source['section']}")

        if doc_info:
            parts.append(" • ".join(doc_info))

        if source.get('score') is not None:
            score_percent = source['score'] * 100
            parts.append(f"🎯 Релевантність: {score_percent:.1f}%")

        if source.get('content'):
            parts.append(f"\n> {source['content'][:300]}...")

    return "\n\n".join(parts)


def main():
    # -----------------------------
    # Session state ініціалізація
    # -----------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    # -----------------------------
    # Sidebar — Documents
    # -----------------------------
    st.sidebar.title("📚 Управління документами")
    mode = st.sidebar.radio("Додати документ з:", ["URL", "Файлу"])

    if mode == "URL":
        url = st.sidebar.text_input("Введіть URL", placeholder="https://example.com/document")
        if st.sidebar.button("➕ Додати URL", disabled=not url):
            if not url.startswith(('http://', 'https://')):
                st.sidebar.error("❌ URL має починатися з http:// або https://")
            else:
                try:
                    with st.spinner("Завантаження..."):
                        r = requests.post(
                            f"{API_BASE}/vector-memory/documents/url",
                            params={"url": url},
                            timeout=60
                        )
                    if r.status_code in [200, 202]:
                        data = r.json()
                        st.sidebar.success(f"✅ {data.get('message', 'Додано')}")
                    else:
                        try:
                            error_detail = r.json().get('detail', 'Невідома помилка')
                        except:
                            error_detail = r.text
                        st.sidebar.error(f"❌ Помилка: {error_detail}")
                except requests.exceptions.Timeout:
                    st.sidebar.error("❌ Перевищено час очікування")
                except Exception as e:
                    st.sidebar.error(f"❌ Не вдалося додати: {str(e)}")

    else:
        file = st.sidebar.file_uploader(
            "Завантажити файл",
            type=["pdf", "docx", "txt", "md", "html"],
        )
        if file and st.sidebar.button("➕ Додати файл"):
            try:
                with st.spinner("Завантаження..."):
                    r = requests.post(
                        f"{API_BASE}/vector-memory/documents/file",
                        files={"file": (file.name, file, file.type)},
                        timeout=120
                    )
                if r.status_code in [200, 202]:
                    data = r.json()
                    st.sidebar.success(f"✅ {data.get('message', 'Файл додається у фоні')}")
                else:
                    try:
                        error_detail = r.json().get('detail', 'Невідома помилка')
                    except:
                        error_detail = r.text
                    st.sidebar.error(f"❌ Помилка: {error_detail}")
            except requests.exceptions.Timeout:
                st.sidebar.error("❌ Перевищено час очікування")
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
        else:
            st.sidebar.warning("⚠️ Reranking вимкнено")

    except Exception as e:
        st.sidebar.warning(f"⚠️ База недоступна: {str(e)}")

    # -----------------------------
    # Clear documents
    # -----------------------------
    if st.sidebar.button("🗑 Очистити всі документи", type="secondary"):
        try:
            r = requests.delete(f"{API_BASE}/vector-memory/clear", timeout=10)
            if r.status_code == 200:
                st.sidebar.success("✅ Базу очищено")
                st.rerun()
            else:
                st.sidebar.error(f"❌ Помилка: {r.status_code}")
        except Exception as e:
            st.sidebar.error(f"❌ Помилка: {str(e)}")

    st.sidebar.divider()

    # -----------------------------
    # New conversation
    # -----------------------------
    if st.sidebar.button("🔁 Нова розмова", type="primary"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.sidebar.success("✅ Розпочато нову розмову")
        st.rerun()

    # ================================
    # Main — Chat
    # ================================
    st.title("💬 RAG чат з джерелами")
    st.caption(f"🔗 Підключено до: {API_BASE}")
    st.caption(f"🆔 Сесія: {st.session_state.session_id[:8]}...")

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
                # Підрахунок джерел за типом
                url_sources = [s for s in msg["sources"] if s.get("source_type") == "url"]
                doc_sources = [s for s in msg["sources"] if s.get("source_type") == "document"]

                source_label = f"📚 Джерела ({len(msg['sources'])})"
                if url_sources and doc_sources:
                    source_label += f" • {len(url_sources)} URL • {len(doc_sources)} документів"
                elif url_sources:
                    source_label += f" • {len(url_sources)} URL"
                elif doc_sources:
                    source_label += f" • {len(doc_sources)} документів"

                with st.expander(source_label):
                    for idx, source in enumerate(msg["sources"], 1):
                        st.markdown(format_source(source, idx))
                        if idx < len(msg["sources"]):
                            st.divider()

    # Chat input
    user_input = st.chat_input("Задайте питання на основі документів...")

    if user_input:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Шукаю відповідь..."):
                try:
                    r = requests.post(
                        f"{API_BASE}/agent/chat",
                        json={
                            "query": user_input,
                            "session_id": st.session_state.session_id
                        },
                        timeout=60
                    )

                    if r.status_code == 200:
                        data = r.json()
                        answer = data.get("answer", "Немає відповіді")
                        sources = data.get("sources", [])
                        rewrite_attempts = data.get("rewrite_attempts", 0)
                        query_rewritten = data.get("query_rewritten")

                        st.markdown(answer)

                        if sources:
                            # Підрахунок джерел за типом
                            url_sources = [s for s in sources if s.get("source_type") == "url"]
                            doc_sources = [s for s in sources if s.get("source_type") == "document"]

                            source_label = f"📚 Джерела ({len(sources)})"
                            if url_sources and doc_sources:
                                source_label += f" • {len(url_sources)} URL • {len(doc_sources)} документів"
                            elif url_sources:
                                source_label += f" • {len(url_sources)} URL"
                            elif doc_sources:
                                source_label += f" • {len(doc_sources)} документів"

                            with st.expander(source_label, expanded=True):
                                for idx, source in enumerate(sources, 1):
                                    st.markdown(format_source(source, idx))
                                    if idx < len(sources):
                                        st.divider()
                        else:
                            st.info("💡 Спробуйте переформулювати питання або додати більше документів")

                        with st.expander("ℹ️ Детальна інформація", expanded=False):
                            if rewrite_attempts > 0:
                                st.write(f"🔄 Запит було переформульовано {rewrite_attempts} раз(и)")
                            if query_rewritten and query_rewritten != user_input:
                                st.write("🔍 Оптимізований запит:")
                                st.code(query_rewritten)
                            st.write(f"📊 Знайдено джерел: {len(sources)}")
                            if sources:
                                url_count = len([s for s in sources if s.get("source_type") == "url"])
                                doc_count = len([s for s in sources if s.get("source_type") == "document"])
                                st.write(f"  • URL: {url_count}")
                                st.write(f"  • Документи: {doc_count}")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })

                    else:
                        try:
                            error_detail = r.json().get('detail', f'HTTP {r.status_code}')
                        except:
                            error_detail = f'HTTP {r.status_code}'

                        error_msg = f"❌ Помилка: {error_detail}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })

                except requests.exceptions.Timeout:
                    error_msg = "❌ Перевищено час очікування відповіді від сервера"
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