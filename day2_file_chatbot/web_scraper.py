# import http.client
# import json
# from dotenv import load_dotenv
# load_dotenv()
# conn = http.client.HTTPSConnection("google.serper.dev")
# payload = json.dumps({
#   "gl": "in"
# })
# headers = {
#   'X-API-KEY': SERPER_API,
#   'Content-Type': 'application/json'
# }
# conn.request("POST", "/search", payload, headers)
# res = conn.getresponse()
# data = res.read()
# print(data.decode("utf-8"))