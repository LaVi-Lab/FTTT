import fastapi

from execution import check_correctness

app = fastapi.FastAPI()

timeout = 3

@app.post("/")
async def predict(req: fastapi.Request):
    req_data = await req.json()
    return check_correctness(req_data["problem"], req_data["completion"], timeout)
