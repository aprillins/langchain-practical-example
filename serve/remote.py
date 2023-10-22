from langserve import RemoteRunnable

remote_chain = RemoteRunnable("http://localhost:8000/chain/")
print(remote_chain.invoke({"language": "italian", "text": "hi"}))


# import requests

# response = requests.post(
#     "http://localhost:8000/chain",
#     json={"language": "italian", "text": "hi"}
# )
# print(response.json())