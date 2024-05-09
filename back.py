from fastapi import HTTPException
from flask import Flask, request, jsonify, Response, redirect, send_from_directory
import os
import threading
import concurrent.futures
import json
import traceback
import re
import requests
from transformers import AutoTokenizer, AutoModel

app = Flask(__name__)

BING_MKT = "en-US"  # 你需要替换为正确的市场代码
BING_SEARCH_V7_ENDPOINT = "https://api.cognitive.microsoft.com/bing/v7.0/search"  # 你需要替换为正确的 API 端点
os.environ["BING_SEARCH_V7_SUBSCRIPTION_KEY"] = "595b3f0ee41e49c38b1bafec91d2359e"
DEFAULT_SEARCH_ENGINE_TIMEOUT = 10  # 你可以根据需要调整超时时间
REFERENCE_COUNT = 10  # 你可以根据需要调整搜索结果的返回数量

class ChatGLM:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("backEnd/model/chatglm3-6b", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("backEnd/model/chatglm3-6b", trust_remote_code=True).float()
        self.history = []
    
    def chat(self, user_input):
        response, self.history = self.model.chat(self.tokenizer, user_input, history=self.history)
        return response

class RAG:
    """
    检索增强生成演示，来自 Lepton AI。
    这是一个最小示例，展示了如何使用 Lepton AI 构建 RAG 引擎。
    它使用搜索引擎根据用户查询获取结果，然后使用 LLM 模型生成答案以及相关问题。
    结果然后存储在 KV 中，以便稍后检索。
    """
    handler_max_concurrency = 16

    def __init__(self):
        """
        初始化 photon 配置。
        """
        self.search_api_key = os.environ["BING_SEARCH_V7_SUBSCRIPTION_KEY"]
        # self.model = os.environ["LLM_MODEL"]
        # self.chat_glm = ChatGLM()
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.handler_max_concurrency * 2
        )
        app.logger.info("Creating KV. May take a while for the first time.")
        # self.kv = KV(
        #     os.environ["KV_NAME"], create_if_not_exists=True, error_if_exists=False
        # )
        # self.should_do_related_questions = to_bool(os.environ["RELATED_QUESTIONS"])

    def search_with_bing(self, query: str, subscription_key: str):
        """
        Search with bing and return the contexts.
        """
        params = {"q": query, "mkt": BING_MKT}
        response = requests.get(
            BING_SEARCH_V7_ENDPOINT,
            headers={"Ocp-Apim-Subscription-Key": subscription_key},
            params=params,
            timeout=DEFAULT_SEARCH_ENGINE_TIMEOUT,
        )
        if not response.ok:
            app.logger.error(f"{response.status_code} {response.text}")
            raise HTTPException(response.status_code, "Search engine error.")
        json_content = response.json()
        try:
            contexts = json_content["webPages"]["value"][:REFERENCE_COUNT]
        except KeyError:
            app.logger.error(f"Error encountered: {json_content}")
            return []
        return contexts

    def generate_answer_with_llm(self, query):
        """
        Generate answer using the local LLM model.
        """
        return self.chat_glm.chat(query)

    @app.route("/", methods=["POST"])
    def query_function(self):
        """
        Query the search engine and returns the response.
        """
        data = request.get_json()
        query = data.get("addAsk", "")
        print(query)
        # search_uuid = data.get("search_uuid", "")
        # generate_related_questions = data.get("generate_related_questions", True)
        contexts = self.search_with_bing(query, self.search_api_key)
        llm_answer = self.generate_answer_with_llm(query)

        response_data = {
            "contexts": contexts,
            "llm_answer": llm_answer,
        }
        print(response_data)
        return jsonify(response_data)



if __name__ == "__main__":
    rag_instance = RAG()
    # rag_instance.init()
    app.run(debug=True, port=5000, host='0.0.0.0')
