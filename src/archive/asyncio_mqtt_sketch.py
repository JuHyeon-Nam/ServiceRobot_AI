import asyncio
import json
import aiomqtt
from fastapi import FastAPI

app = FastAPI()

# [개선된 구조] Non-blocking 비동기 데이터 처리 파이프라인
async def process_sensor_data(msg_payload):
    # 1. 고주파 센서 데이터 파싱
    sensor_data = json.loads(msg_payload)
    
    # 2. 동기식 블로킹(0.2초 지연) 원인이었던 DB 적재와 AI 추론을 분리
    # asyncio.create_task를 통해 병렬(Concurrency) 처리로 전환
    db_task = asyncio.create_task(insert_timeseries_db_async(sensor_data))
    ai_task = asyncio.create_task(run_autoencoder_anomaly_detection(sensor_data))
    
    # 두 작업이 서로를 기다리지 않고 동시 실행되도록 묶음 (병목 해결 핵심)
    await asyncio.gather(db_task, ai_task)
    
    # 3. 진단 결과를 3D 디지털 트윈 대시보드 웹소켓으로 지연 없이 브로드캐스트
    anomaly_score = ai_task.result()
    await websocket_manager.broadcast({"agv_id": sensor_data["id"], "score": anomaly_score})


# [MQTT 수신부]
async def mqtt_subscriber_loop():
    async with aiomqtt.Client("mqtt://factory-broker.local") as client:
        await client.subscribe("factory/agv/high_freq_sensors")
        
        async for message in client.messages:
            # 수신 즉시 이벤트 루프를 멈추지 않고 백그라운드 태스크로 넘김
            asyncio.create_task(process_sensor_data(message.payload))

@app.on_event("startup")
async def startup_event():
    # FastAPI 서버 시작 시 MQTT 구독자 비동기 루프 실행
    asyncio.create_task(mqtt_subscriber_loop())