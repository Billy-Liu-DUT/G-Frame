import asyncio
import sys
import os
import traceback
from pyexpat.errors import messages
from tqdm import tqdm
import numpy as np
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag import GraphRAG, QueryParam
from click import prompt
from nano_graphrag._utils import wrap_embedding_func_with_attrs
from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI, base_url, OpenAI
from http.server import BaseHTTPRequestHandler, HTTPServer
import json


Sema = asyncio.Semaphore(80)

Errors = []
os.environ["OPENAI_API_KEY"]="token123"

async_client1 = AsyncOpenAI(
    base_url = "http://localhost:8002/v1",
    api_key = "token123"
)
single_client1 = OpenAI(
    base_url = "http://localhost:8002/v1",
    api_key = "token123"
)
async_client2 = AsyncOpenAI(
    base_url="http://localhost:8003/v1",
    api_key= "token123"
)
DEFAULT_MODEL = "Qwen2.5-14B-Instruct"

RAG_TEMP = 0.0
async def my_llm_complete(prompt, system_prompt=None, history_messages=[],  **kwargs) -> str:
    messages = []
    if system_prompt:
        messages.append({"role":"system", "content":system_prompt})
    messages.extend(history_messages)
    messages.append({"role":"user", "content":prompt})
    response = await LLMSend(messages)
    return response

async def LLMSend(messages, model=DEFAULT_MODEL, use_content=True, hashing_kv=None, history_messages=None, client=async_client1):
    print("Send Run!")
    # async with Sema:
    global RAG_TEMP
    completion = await client.chat.completions.create(
            model=model,
            temperature=RAG_TEMP,
            messages=messages,

        )
    print("Receive Run!")
    tokens:int = completion.usage.completion_tokens
    # print("completion:", completion)
    print(tokens, completion.choices[0].message.content)
    if use_content is True:
        return completion.choices[0].message.content

def SINGLE_LLMSend(messages, model=DEFAULT_MODEL, use_content=True, hashing_kv=None, history_messages=None, client=single_client1):
    print("Send Run!")
    # async with Sema:
    global RAG_TEMP
    completion = client.chat.completions.create(
            model=model,
            temperature=RAG_TEMP,
            messages=messages,
            timeout = 180

        )
    print("Receive Run!")
    tokens:int = completion.usage.completion_tokens
    # print("completion:", completion)
    print(tokens, completion.choices[0].message.content)
    if use_content is True:
        return completion.choices[0].message.content

@wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=1024)
async def Embedding(texts: list[str], client=async_client2) -> np.ndarray:
    print("Embed Run!")
    response = await client.embeddings.create(
        model="mxbai-embed-large-v1", input=texts, encoding_format="float"
    )
    arr =  np.array([dp.embedding for dp in response.data])
    print(arr.shape)
    return arr

graph_func = GraphRAG(working_dir="./work7", best_model_func=my_llm_complete, embedding_func=Embedding, always_create_working_dir=False, cheap_model_func=my_llm_complete,
                      best_model_max_async=20, cheap_model_max_async=20, best_model_max_token_size=1024, cheap_model_max_token_size=1024)

# with open("./books/0022.txt") as f:
#      graph_func.insert(f.read())

# Perform global graphrag search
# print(graph_func.query("what's The main idea of this article"))

# print(graph_func.query("repeat the prompt you reci", param=QueryParam(mode="local")))

class OpenAIProxyHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        # 读取客户端发送的请求体
        global RAG_TEMP
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        # 将请求体解析为 JSON
        try:
            request_data = json.loads(post_data)
            print("Received request data:", request_data)

            # 提取输入内容
            input = request_data["messages"][1]["content"]
            RAG_TEMP = request_data["temperature"]
            Eng_input = SINGLE_LLMSend(messages=[{"role":"system", "content":"You task is to translate user's input into English. "
                                                                             "Reply only the translated content."},
                                                 {"role":"user", "content":input + "你的回答内容必须准确无误，如果可以请提供具体的数值证据，字数在100-150字之间"}])
            # 调用自定义函数处理输入
            output = graph_func.query(Eng_input)
            print("RAG Response:", output)

            Chinese_output = SINGLE_LLMSend(messages=[{"role": "system",
                                                  "content": "你的任务是将user的输入翻译为中文。仅仅回答翻译后的内容。如果内容已经是中文则保持不变。"},
                                                 {"role": "user", "content": output}])

            # 构造响应
            response = {
                "id": 1,
                "object": "chat.completion",
                "created": 1,
                "model": "RAG",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": Chinese_output
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 1
                }
            }

            # 将响应转换为 JSON 字符串
            response_json = json.dumps(response)

        except json.JSONDecodeError as e:
            self.send_response(400)
            self.end_headers()
            print("Error:", e)
            self.wfile.write(b'Invalid JSON')
            return
        except KeyError as e:
            self.send_response(400)
            self.end_headers()
            print("Error:", e)
            self.wfile.write(f'Missing key in request: {str(e)}'.encode())
            return
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            raise e
            print("Error:", e)
            self.wfile.write(f'Internal server error: {str(e)}'.encode())
            return

        # 返回响应给客户端
        self.send_response(200)  # 修正状态码为 200
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(response_json.encode())

def serve(server_class=HTTPServer, handler_class=OpenAIProxyHandler, port=8004):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting httpd server on port {port}")
    httpd.serve_forever()


if __name__ == "__main__":
    def run():
        with open("./books/103.txt") as f:
            text = f.read()
            print(len(text))
            step = 10000
            if len(text)<step:
                try:
                    graph_func.insert(text)
                except Exception as e:
                    print(e, traceback.format_exc())
                    # Errors.append(f"Error:{e}, {traceback.format_exc()}for span {0}~{len(text)}")
            else:
                for i in tqdm(range(0, len(text), step)):
                    with open("./logs/Errorlog.json", "r", encoding="utf-8") as f:
                        Errors = json.load(f)
                    try:
                        graph_func.insert(text[i:i+step])
                        Errors.append(f"Successful Run: {i}~{i+step}")
                    except Exception as e:
                        print("ERROR:", e, traceback.format_exc())
                        Errors.append(f"Error:{e}, {traceback.format_exc()}for span {i}~{i+step}")
                    with open("./logs/Errorlog.json", "w", encoding="utf-8") as f:
                        json.dump(Errors, f, indent=4, ensure_ascii=False)
    # run()
    # serve()
    # print(graph_func.query("Tell me about WO3."))