
import os
from pathlib import Path
from typing import TypedDict

import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from openai import OpenAI
from serpapi import GoogleSearch
from langgraph.graph import StateGraph, START, END

# -----------------------------
# 1. 기본 설정
# -----------------------------
st.set_page_config(
    page_title="제품 설명서 챗봇",
    page_icon="📘"
)

st.title("📘 제품 설명서 챗봇")
st.write("PDF 설명서를 검색하거나 웹에서 정보를 찾아 답변받을 수 있어요.")

PROJECT_PATH = Path(__file__).parent
PDF_PATH = PROJECT_PATH / "homabot_manual.pdf"

# .env 파일에서 API 키 불러오기
load_dotenv(PROJECT_PATH / ".env", override=False)

def get_secret(name):
    # 배포 환경은 st.secrets, 로컬은 .env 순서로 키를 찾음
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return (value or os.getenv(name) or "").strip() or None

OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
SERPAPI_API_KEY = get_secret("SERPAPI_API_KEY") or get_secret("SERPAPI_KEY")

# OpenAI 클라이언트 준비
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None

# -----------------------------
# 2. PDF 읽기 및 텍스트 분할
# -----------------------------
@st.cache_data
def load_pdf_chunks(pdf_file):
    reader = PdfReader(str(pdf_file))
    chunks = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        for start in range(0, len(text), 500):
            chunk_text = text[start:start + 500]

            if chunk_text.strip():
                chunks.append({
                    "page": page_number,
                    "text": chunk_text
                })

    return chunks

if not PDF_PATH.exists():
    st.error(f"PDF 파일을 찾을 수 없습니다: {PDF_PATH}")
    st.stop()

chunks = load_pdf_chunks(PDF_PATH)

# -----------------------------
# 3. LangGraph 상태 정의
# -----------------------------
class ChatState(TypedDict):
    question: str
    search_mode: str
    context: str
    answer: str

# -----------------------------
# 4. PDF 검색 노드
# -----------------------------
def pdf_search(state: ChatState):
    question_words = [
        word.strip("?!.,:;()[]\"'")
        for word in state["question"].split()
        if len(word.strip("?!.,:;()[]\"'")) >= 2
    ]

    scored_chunks = []

    for chunk in chunks:
        text = chunk["text"].lower()

        score = sum(
            1 for word in question_words
            if word.lower() in text
        )

        if score > 0:
            scored_chunks.append((score, chunk))

    scored_chunks.sort(
        key=lambda item: item[0],
        reverse=True
    )

    selected = [item[1] for item in scored_chunks[:5]]

    if not selected:
        return {
            "context": "",
            "answer": "질문과 관련된 내용을 PDF에서 찾지 못했습니다."
        }

    context = "\n\n".join(
        f"PDF {item['page']}페이지:\n{item['text']}"
        for item in selected
    )

    return {"context": context, "answer": ""}

# -----------------------------
# 5. 웹 검색 노드
# -----------------------------
def web_search(state: ChatState):
    if not SERPAPI_API_KEY:
        return {
            "context": "",
            "answer": "SerpAPI 키가 설정되지 않았습니다. .env 파일을 확인해 주세요."
        }

    params = {
        "engine": "google",
        "q": state["question"],
        "api_key": SERPAPI_API_KEY
    }

    try:
        results = GoogleSearch(params).get_dict()
    except Exception as e:
        return {
            "context": "",
            "answer": f"웹 검색 중 오류가 발생했습니다: {e}"
        }

    if results.get("error"):
        return {
            "context": "",
            "answer": f"웹 검색 서비스 오류: {results['error']}"
        }

    items = results.get("organic_results", [])[:5]

    if not items:
        return {
            "context": "",
            "answer": "웹 검색 결과를 찾지 못했습니다."
        }

    context = "\n\n".join(
        f"제목: {item.get('title', '')}\n"
        f"링크: {item.get('link', '')}\n"
        f"요약: {item.get('snippet', '')}"
        for item in items
    )

    return {"context": context, "answer": ""}

# -----------------------------
# 6. 검색 방식 선택
# -----------------------------
def choose_search(state: ChatState):
    if state["search_mode"] == "PDF 설명서 검색":
        return "pdf_search"
    return "web_search"

# -----------------------------
# 7. OpenAI 답변 생성 노드
# -----------------------------
def generate_answer(state: ChatState):
    # 검색 단계에서 답변 오류 메시지를 만든 경우 그대로 반환
    if state["answer"]:
        return {}

    if not state["context"].strip():
        return {"answer": "참고할 검색 결과가 없어 답변을 만들 수 없습니다."}

    if client is None:
        return {"answer": "OpenAI API 키가 설정되지 않았습니다. .env 파일을 확인해 주세요."}

    if state["search_mode"] == "PDF 설명서 검색":
        source_instruction = (
            "PDF 내용에 근거해 답변하고, 사용한 PDF 페이지 번호를 표시해라."
        )
    else:
        source_instruction = (
            "웹 검색 결과를 근거로 답변하고, 관련 링크를 함께 안내해라."
        )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "너는 정확하고 친절한 한국어 안내 챗봇이다. "
                        "제공된 자료를 바탕으로 답변하고, "
                        "자료에 없는 사실은 지어내지 마라. "
                        + source_instruction
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"질문: {state['question']}\n\n"
                        f"참고 자료:\n{state['context']}"
                    )
                }
            ],
            temperature=0
        )

        answer = response.choices[0].message.content
        return {"answer": answer or "답변을 생성하지 못했습니다."}

    except Exception as e:
        return {"answer": f"AI 답변 생성 중 오류가 발생했습니다: {e}"}

# -----------------------------
# 8. LangGraph 구성
# -----------------------------
builder = StateGraph(ChatState)

builder.add_node("pdf_search", pdf_search)
builder.add_node("web_search", web_search)
builder.add_node("generate_answer", generate_answer)

builder.add_conditional_edges(
    START,
    choose_search,
    {
        "pdf_search": "pdf_search",
        "web_search": "web_search"
    }
)

builder.add_edge("pdf_search", "generate_answer")
builder.add_edge("web_search", "generate_answer")
builder.add_edge("generate_answer", END)

graph = builder.compile()

# -----------------------------
# 9. 사용자 화면
# -----------------------------
st.caption(f"PDF 준비 완료: {len(chunks)}개 텍스트 조각")

search_mode = st.radio(
    "검색 방식을 선택하세요.",
    ["PDF 설명서 검색", "웹 검색"],
    horizontal=True
)

question = st.text_input(
    "질문을 입력하세요.",
    placeholder="예: 제품 설치 방법을 알려줘"
)

if st.button("질문하기", type="primary"):
    if not question.strip():
        st.warning("질문을 먼저 입력해 주세요.")

    else:
        with st.spinner("검색하고 답변을 작성하고 있어요..."):
            initial_state = {
                "question": question.strip(),
                "search_mode": search_mode,
                "context": "",
                "answer": ""
            }

            try:
                result = graph.invoke(initial_state)

                st.subheader("💬 답변")
                st.write(result["answer"])

                if result["context"].strip():
                    with st.expander("📄 참고한 검색 자료 보기"):
                        st.text(result["context"])

            except Exception as e:
                st.error(f"실행 중 오류가 발생했습니다: {e}")
