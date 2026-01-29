import streamlit as st
import requests
import os
import uuid

API_BASE = os.getenv("API_BASE", "http://localhost:8000")


def format_source(source: dict, idx: int) -> str:
    source_type = source.get("source_type", "document")
    parts = []

    if source_type == "url":
        parts.append(f"**[{idx}] 🔗 URL**")
        parts.append(f"[{source['source']}]({source['source']})")
    else:
        parts.append(f"**[{idx}] 📄 {source['source']}**")
        info = []
        if source.get("page") is not None:
            info.append(f"Сторінка {source['page']}")
        if source.get("section"):
            info.append(f"Розділ: {source['section']}")
        if info:
            parts.append(" • ".join(info))

    if source.get("score") is not None:
        parts.append(f"🎯 Релевантність: {source['score'] * 100:.1f}%")

    if source.get("content"):
        parts.append(f"\n> {source['content'][:300]}...")

    return "\n\n".join(parts)


def main():
    # ==============================
    # Session state
    # ==============================
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    # ==============================
    # Sidebar — Documents
    # ==============================
    st.sidebar.title("📚 Управління документами")
    mode = st.sidebar.radio("Додати документ з:", ["URL", "Файлу"])

    if mode == "URL":
        url = st.sidebar.text_input("Введіть URL")
        if st.sidebar.button("➕ Додати URL", disabled=not url):
            try:
                r = requests.post(
                    f"{API_BASE}/vector-memory/documents/url",
                    params={"url": url},
                    timeout=60,
                )
                if r.ok:
                    st.sidebar.success("✅ URL додано")
                else:
                    st.sidebar.error(r.text)
            except Exception as e:
                st.sidebar.error(f"❌ Помилка: {str(e)}")
    else:
        file = st.sidebar.file_uploader(
            "Завантажити файл",
            type=["pdf", "docx", "txt", "md", "html"],
        )
        if file and st.sidebar.button("➕ Додати файл"):
            try:
                r = requests.post(
                    f"{API_BASE}/vector-memory/documents/file",
                    files={"file": (file.name, file, file.type)},
                    timeout=120,
                )
                if r.ok:
                    st.sidebar.success("✅ Файл додано")
                else:
                    st.sidebar.error(r.text)
            except Exception as e:
                st.sidebar.error(f"❌ Помилка: {str(e)}")

    st.sidebar.divider()

    # ==============================
    # New conversation
    # ==============================
    if st.sidebar.button("🔁 Нова розмова"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    # ==============================
    # Chat UI
    # ==============================
    st.title("💬 RAG чат з джерелами")
    st.caption(f"🔗 API: {API_BASE}")
    st.caption(f"🆔 Session: {st.session_state.session_id[:8]}...")

    # Рендер попередніх повідомлень
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ==============================
    # Input
    # ==============================
    user_input = st.chat_input("Задайте питання на основі документів...")

    if not user_input:
        return

    # Додаємо повідомлення користувача
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ==============================
    # Відправка запиту асистенту
    # ==============================
    with st.chat_message("assistant"):
        with st.spinner("🤔 Шукаю відповідь..."):
            try:
                r = requests.post(
                    f"{API_BASE}/agent/chat",
                    json={"query": user_input, "session_id": st.session_state.session_id},
                    timeout=60,
                )
                if not r.ok:
                    st.error(r.text)
                    return

                data = r.json()
                answer = data.get("answer", "")
                sources = data.get("sources", [])

                # Відповідь завжди на першому плані
                st.markdown(f"**💡 Відповідь:** {answer}")

                # Зберігаємо повідомлення
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as e:
                st.error(f"❌ Помилка: {str(e)}")
                return

    # ==============================
    # Collapsed sources block (усі джерела)
    # ==============================
    all_sources = []
    for msg in st.session_state.messages:
        if msg["role"] == "assistant" and msg.get("sources"):
            all_sources.extend(msg["sources"])

    if all_sources:
        with st.expander(f"📚 Джерела ({len(all_sources)})", expanded=False):
            for i, src in enumerate(all_sources, 1):
                st.markdown(format_source(src, i))
                if i < len(all_sources):
                    st.divider()


if __name__ == "__main__":
    st.set_page_config(
        page_title="RAG чат",
        page_icon="💬",
        layout="wide",
    )
    main()
