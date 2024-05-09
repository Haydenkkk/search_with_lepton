import concurrent.futures
import glob
import json
import os
import re
import threading
import requests
import traceback
from typing import Generator, Optional
from fastapi import HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from loguru import logger
import redis
from leptonai.photon import Photon, StaticFiles

os.environ["BING_SEARCH_V7_SUBSCRIPTION_KEY"] = "595b3f0ee41e49c38b1bafec91d2359e"
os.environ["LEPTON_WORKSPACE_TOKEN"] = "bfle763t:xwfqodgbg11yhn56blr1tkqv3hzm37n0"
os.environ["LLM_MODEL"] = "backEnd/model/chatglm3-6b"
BING_SEARCH_V7_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"
BING_MKT = "zh-CN"

REFERENCE_COUNT = 8

DEFAULT_SEARCH_ENGINE_TIMEOUT = 10

_default_query = "Who said 'live long and prosper'?"

_rag_query_text = """
你是由北京邮电大学Hayden构建的一款大型语言AI助手。你将获得一个用户问题，请写出简洁、准确的答案。你将获得一组与问题相关的上下文，每个上下文都以[[citation:x]]开头，其中x是一个数字。如果适用，请在每个句子末尾使用上下文并引用上下文。

你的答案必须正确、准确，并由专家以公正、专业的语气撰写。请将答案限制在1024个标记以内。不要提供与问题无关的任何信息，并且不要重复。如果给定的上下文不能提供足够的信息，请说“关于”后跟相关主题，信息不足。

请以 [citation:x] 的格式引用带有参考号的上下文。如果一句话来自多个上下文，请列出所有适用的引用，比如 [citation:3][citation:5]。除了代码和特定名称和引用外，您的答案必须用与问题相同的语言撰写。

以下是一组上下文：

{context}

请记住，不要盲目地逐字重复上下文。以下是用户问题：
"""

_more_questions_prompt = """
您是一位乐于助人的助手，可以根据用户的原始问题和相关上下文帮助用户提出相关问题。请识别可作为后续问题的值得关注的主题，并编写每个问题不超过20个字。请确保在后续问题中包含具体信息，如事件、名称、地点，以便它们可以独立提出。例如，如果原始问题问及“曼哈顿计划”，在后续问题中不要只说“该项目”，而要使用全名“曼哈顿计划”。您的相关问题必须与原始问题使用相同的语言。

以下是问题的上下文：

{context}

请记住，根据原始问题和相关上下文，建议三个这样的后续问题。请不要重复原始问题。每个相关问题的长度不得超过20个字。以下是原始问题：
"""


def search_with_bing(query: str, subscription_key: str):
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
        logger.error(f"{response.status_code} {response.text}")
        raise HTTPException(response.status_code, "Search engine error.")
    json_content = response.json()
    try:
        contexts = json_content["webPages"]["value"][:REFERENCE_COUNT]
    except KeyError:
        logger.error(f"Error encountered: {json_content}")
        return []
    return contexts


class RAG(Photon):
    extra_files = glob.glob("ui/**/*", recursive=True)  # 额外的文件，用于 UI。
    handler_max_concurrency = 16

    def local_client(self):
        from zhipuai import ZhipuAI
        thread_local = threading.local()
        try:
            return thread_local.client
        except AttributeError:
            thread_local.client = ZhipuAI(
                api_key='8767d299509be33f14d33b9bdeb7c798.D5FIYGXvLis4RYVk')
            return thread_local.client

    def init(self):
        self.model = "glm-4"
        self.search_api_key = os.environ["BING_SEARCH_V7_SUBSCRIPTION_KEY"]
        self.search_function = lambda query: search_with_bing(
            query,
            self.search_api_key,
        )
        # 是否应该生成相关问题。
        self.should_do_related_questions = True
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.handler_max_concurrency * 2
        )
        self.redis_client = redis.Redis(
            host='localhost', port=6379, db=0, decode_responses=True
        )

    def get_related_questions(self, query, contexts):
        try:
            response = self.local_client().chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": _more_questions_prompt.format(
                            context="\n\n".join([c["snippet"]
                                                for c in contexts])
                        ),
                    },
                    {
                        "role": "user",
                        "content": query,
                    },
                ],
                max_tokens=512,
            )
            related = response.choices[0].message.content.split("\n")
            return related[:3]
        except Exception as e:
            logger.error(
                "encountered error while generating related questions:"
                f" {e}\n{traceback.format_exc()}"
            )
            return []

    def _raw_stream_response(
        self, contexts, llm_response, related_questions_future
    ) -> Generator[str, None, None]:
        """
        A generator that yields the raw stream response. You do not need to call
        this directly. Instead, use the stream_and_upload_to_kv which will also
        upload the response to KV.
        """
        # First, yield the contexts.
        yield json.dumps(contexts, ensure_ascii=False)
        yield "\n\n__LLM_RESPONSE__\n\n"
        # Second, yield the llm response.
        if not contexts:
            # Prepend a warning to the user
            yield (
                "(The search engine returned nothing for this query. Please take the"
                " answer with a grain of salt.)\n\n"
            )
        for chunk in llm_response:
            if chunk.choices:
                yield chunk.choices[0].delta.content or ""
        # Third, yield the related questions. If any error happens, we will just
        # return an empty list.
        if related_questions_future is not None:
            related_questions = related_questions_future.result()
            # print(related_questions)
            try:
                json_questions = [{"question": question}
                                  for question in related_questions]

                result = json.dumps(json_questions, ensure_ascii=False)

            except Exception as e:
                logger.error(
                    f"encountered error: {e}\n{traceback.format_exc()}")
                result = "[]"
            yield "\n\n__RELATED_QUESTIONS__\n\n"
            yield result

    def stream_and_upload_to_kv(
        self, contexts, llm_response, related_questions_future, search_uuid
    ) -> Generator[str, None, None]:
        """
        Streams the result and uploads to KV.
        """
        # First, stream and yield the results.
        all_yielded_results = []
        for result in self._raw_stream_response(
            contexts, llm_response, related_questions_future
        ):
            all_yielded_results.append(result)
            yield result
        # Second, upload to KV. Note that if uploading to KV fails, we will silently
        # ignore it, because we don't want to affect the user experience.
        # _ = self.executor.submit(self.kv.put, search_uuid, "".join(all_yielded_results))

    @Photon.handler(method="POST", path="/query")
    async def query_function(
        self,
        query: str,
        search_uuid: str,
        generate_related_questions: Optional[bool] = True,
    ) -> StreamingResponse:

        query = query or _default_query
        query = re.sub(r"\[/?INST\]", "", query)
        contexts = self.search_function(query)

        system_prompt = _rag_query_text.format(
            context="\n\n".join(
                [f"[[citation:{i+1}]] {c['snippet']}" for i,
                    c in enumerate(contexts)]
            )
        )
        try:
            client = self.local_client()
            llm_response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                max_tokens=1024,
                stream=True,
                temperature=0.1,
            )

            if self.should_do_related_questions and generate_related_questions:
                related_questions_future = self.executor.submit(
                    self.get_related_questions, query, contexts
                )
            else:
                related_questions_future = None
        except Exception as e:
            logger.error(f"encountered error: {e}\n{traceback.format_exc()}")
            return HTMLResponse("Internal server error.", 503)

        # 将结果存入 Redis
        # all_yielded_results = list(self._raw_stream_response(contexts, llm_response, related_questions_future))

        return StreamingResponse(
            self.stream_and_upload_to_kv(
                contexts, llm_response, related_questions_future, search_uuid
            ),
            media_type="text/html",
        )

    @Photon.handler(mount=True)
    def ui(self):
        return StaticFiles(directory="ui")

    @Photon.handler(method="GET", path="/")
    def index(self) -> RedirectResponse:
        """
        Redirects "/" to the ui page.
        """
        return RedirectResponse(url="/ui/index.html")


if __name__ == "__main__":
    rag = RAG()
    rag.launch()
