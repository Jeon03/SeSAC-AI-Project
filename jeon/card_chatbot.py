
import json
import os
import time
import torch
from typing import List, Any, Dict
from dotenv import load_dotenv

# 필수 라이브러리
from openai import OpenAI
import cohere
from sentence_transformers import SentenceTransformer
import chromadb

# LangChain 관련
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereRerank

# 패키지 경로 에러 방지를 위한 처리
try:
    from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
except ImportError:
    from langchain.retrievers import ContextualCompressionRetriever

from langchain_core.retrievers import BaseRetriever

# 1. 환경 변수 로드
load_dotenv()

# ==========================================
# [CUSTOM CLASS] 1. 임베딩 및 리트리버 설정
# ==========================================

class GemmaEmbeddings:
    """Gemma-300m 전용 임베딩 클래스"""
    def __init__(self, model_path: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_path, device=device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode([f"title: none | text: {text}" for text in texts]).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(f"task: search result | query: {text}").tolist()

class CardANDRetriever(BaseRetriever):
    """멀티 의도 교집합(AND) 검색 및 전체 데이터 복원 리트리버"""
    vectorstore: Any
    card_map: Dict
    intent_extractor: Any
    search_depth: int = 200

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # 1. 의도 추출 (LLM 활용)
        keywords_json = self.intent_extractor.invoke({"question": query})
        try:
            search_intents = json.loads(keywords_json.replace("```json", "").replace("```", ""))
        except:
            search_intents = [query]

        # 2. 각 의도별 검색 수행
        intent_scores = []
        for intent in search_intents:
            results = self.vectorstore.similarity_search_with_relevance_scores(intent, k=self.search_depth)
            current_hits = {str(doc.metadata['card_id']): score for doc, score in results}
            intent_scores.append(current_hits)

        # 3. 교집합(AND) 필터링
        common_ids = set(intent_scores[0].keys())
        for hits in intent_scores[1:]:
            common_ids &= set(hits.keys())

        if not common_ids:
            common_ids = set(list(intent_scores[0].keys())[:20])

        # 4. 전체 데이터 복원 (검색은 idx로, 답변 데이터는 원본 전체로)
        final_docs = []
        for c_id in common_ids:
            total_score = sum(intent_map[c_id] for intent_map in intent_scores if c_id in intent_map)
            card = self.card_map.get(c_id) # 여기서 모든 정보가 담긴 Dict를 가져옴
            if not card: continue

            # LLM에게 전달할 풍부한 컨텍스트 생성
            # 여러 카테고리의 content를 합쳐서 전달
            full_benefits = " / ".join(card['full_details'])

            final_docs.append(Document(
                page_content=full_benefits,
                metadata={
                    "total_score": total_score,
                    "card_id": c_id,
                    "name": card['name'],
                    "corp": card['corp'],
                    "annual_fee": card['metadata'].get('annual_fee'),
                    "min_performance": card['metadata'].get('min_performance'),
                    "structured": card['structured'] # ai_structured 전체 포함
                }
            ))

        return sorted(final_docs, key=lambda x: x.metadata['total_score'], reverse=True)

# ==========================================
# [CORE] 2. 메인 RAG 클래스
# ==========================================

class CardConciergeRAG:
    def __init__(self, model_path, db_path, data_path):
        self.embeddings = GemmaEmbeddings(model_path)
        self.data_path = data_path

        self.card_map = self._setup_card_map()
        self.vectorstore = Chroma(
            collection_name="card_benefits",  
            persist_directory=db_path, 
            embedding_function=self.embeddings
        )
        self.chain = self._build_chain()

    def _setup_card_map(self):
        """원본 JSON을 ID 기반으로 맵핑 (모든 카테고리 정보 통합)"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        mapping = {}
        for item in data:
            c_id = str(item['metadata']['card_id'])
            if c_id not in mapping:
                mapping[c_id] = {
                    "name": item['metadata']['card_name'], 
                    "corp": item['metadata']['corp'],
                    "metadata": item['metadata'],
                    "full_details": [item['content']], 
                    "structured": item['ai_structured']
                }
            else:
                mapping[c_id]["full_details"].append(item['content'])
        return mapping

    def _build_chain(self):
        # 의도 추출기
        intent_prompt = ChatPromptTemplate.from_template("""
        사용자의 질문을 분석하여 카드 검색에 가장 적합한 2~3개의 검색 문장을 생성하세요.
        형식: ["문장1", "문장2"] (JSON 리스트로만 응답)
        질문: {question}
        """)
        intent_extractor = (
            intent_prompt 
            | ChatOpenAI(model="gpt-4.1-mini", temperature=0.1, model_kwargs={"top_p": 0.9}) 
            | StrOutputParser()
        )

        # 리트리버 + 리랭커
        base_retriever = CardANDRetriever(
            vectorstore=self.vectorstore, 
            card_map=self.card_map, 
            intent_extractor=intent_extractor,
            search_depth=200
        )
        compressor = CohereRerank(model="rerank-v3.5", top_n=3)
        retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)

        # 최종 답변 프롬프트 (전문가 페르소나)
        prompt = ChatPromptTemplate.from_template("""
        당신은 대한민국 최고의 '신용/체크카드 추천 전문가, Gemma-Bot'입니다. 
        제공된 [카드 데이터]를 분석하여 전문적이면서도 다정하게 답변하세요.

        사용자 질문: {question}
        [카드 데이터]: {context}

        [답변 작성 가이드]
        1. **전문가 인사**: 소비 니즈를 정확히 분석했음을 알리며 신뢰감 있게 시작하세요.
        2. **순위 고수**: Top 1, 2, 3 순서를 절대 바꾸지 마세요.
        3. **상세 분석**: 1위 카드의 혜택(수치, 한도, 실적)을 구체적인 예시와 함께 상세히 설명하세요.
           분석 마지막에는 "🔗 [카드 상세정보 확인하기](상세 링크)"를 포함하세요.
        4. **비서의 조언**: 혜택 제외 항목(benefit_exclusions)과 실적 제외 항목(performance_exclusions)을 참고하여 주의사항을 예리하게 조언하세요.
        5. **톤앤매너**: 표(Table) 사용 금지, 불렛 포인트 활용, 답변은 핵심 위주로 명확하게 작성하세요.
        """)

        # 데이터 구조화 함수 (검색 결과 -> LLM용 텍스트)
        def format_docs(docs):
            formatted = []
            for i, doc in enumerate(docs):
                m = doc.metadata
                s = m.get('structured', {})
                info = (
                    f"### [추천 순위 {i+1}위] {m.get('name')} ({m.get('corp')})\n"
                    f"- 연회비: {m.get('annual_fee', '정보 없음')}\n"
                    f"- 전월 실적 기준: {m.get('min_performance', '정보 없음')}원\n"
                    f"- 상세 링크: https://www.card-gorilla.com/card/detail/{m.get('card_id')}\n"
                    f"- 혜택 요약: {s.get('summary', '정보 없음')}\n"
                    f"- 혜택 제외: {', '.join(s.get('benefit_exclusions', ['정보 없음']))}\n"
                    f"- 실적 제외: {', '.join(s.get('performance_exclusions', ['정보 없음']))}\n"
                    f"- 추가 정보: {s.get('additional_info', '정보 없음')}\n"
                    f"- 상세 혜택 데이터: {doc.page_content}\n"
                )
                formatted.append(info)
            return "\n\n".join(formatted)

        return (
            {
                "context": retriever | format_docs, 
                "question": RunnablePassthrough()
            }
            | prompt 
            | ChatOpenAI(
                model="gpt-4.1-mini", 
                temperature=0.1, 
                model_kwargs={"top_p": 0.1},
                max_tokens=1500
            ) 
            | StrOutputParser()
        )

    def ask(self, query):
        return self.chain.invoke(query)

# ==========================================
# [CLI] 3. CMD 인터페이스
# ==========================================

def run_chatbot():
    print("\n🚀 카드 추천 전문가 Gemma-Bot 시스템 가동 중...")

    try:
        concierge = CardConciergeRAG(
            model_path='./models/gemma-300m-4080super-extreme',
            db_path='./data/chroma_db',
            data_path='./data/FINAL_MASTER_DATA_FIXED_7757.json'
        )
        print("✅ 상담 준비 완료!")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return

    print("\n" + "="*60)
    print("   💳 대한민국 최고의 카드 추천 전문가, Gemma-Bot   ")
    print("      (종료하시려면 '종료' 또는 'q'를 입력하세요)      ")
    print("="*60)

    while True:
        user_input = input("\n[👤 질문]: ").strip()
        if user_input.lower() in ['종료', 'q', 'exit']: break
        if not user_input: continue

        print("\n[🤖 Gemma-Bot]: 최적의 카드를 분석 중입니다...", end="", flush=True)
        start_time = time.time()

        try:
            response = concierge.ask(user_input)
            print(f"\r[🤖 Gemma-Bot] ({time.time() - start_time:.2f}초):")
            print("-" * 60 + "\n" + response + "\n" + "-" * 60)
        except Exception as e:
            print(f"\n❌ 오류: {e}")

if __name__ == "__main__":
    run_chatbot()
