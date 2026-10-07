from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import os

API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("BASE_URL")

# ── 권장 형태 ──
# - max_tokens 기본값을 2048로 늘림: Gemini 같은 추론(thinking) 모델은
#   추론 토큰도 max_tokens에서 차감되므로 512로는 답변이 중간에 잘림.
# - reasoning_effort를 선택 인자로 추가: 값을 준 경우에만 전달해서
#   추론을 지원하지 않는 모델에는 영향이 없도록 함.
#
def get_model(
    model: str = "gpt-5.4-mini",
    temperature: float = 0,
    max_tokens: int = 512,
    reasoning_effort: str | None = None,  # 예: "minimal", "low", "medium", "high"
):
    kwargs = {}
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    
    return ChatOpenAI(
        model=model,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=temperature,
        use_responses_api=False,  # base url로 할 때는 이부분 넣어야 함.(MonoRouter 사용)
        max_tokens=max_tokens,
        **kwargs
    )

# 사용 예:
# gemini = get_model("gemini-3.8-flash", reasoning_effort="minimal")
# claude = get_model("claude-haiku-4.5")


def embedding_model():
    embedding_model = "text-embedding-3-small"
    embeddings = OpenAIEmbeddings(
        api_key=API_KEY,
        base_url=BASE_URL,
        # use_responses_api=False,  # base url로 할 때는 이부분 넣어야 함.(MonoRouter 사용)
        model=embedding_model
        )
    # print(embeddings)
    return embeddings