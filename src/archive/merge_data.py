import pandas as pd
import json
import os
from pathlib import Path
from tqdm import tqdm

# 1. 경로 설정
base_path = Path(r"C:\Users\SSAFY\Desktop\42.실내공간 유지관리 서비스 로봇 데이터\3.개방데이터\1.데이터\extracted_data")

def merge_robot_data():
    all_data = []
    # TL_ 또는 VL_ 폴더 내의 json만 수집
    json_files = [str(p) for p in base_path.rglob("*.json") if p.parent.name.startswith(("TL_", "VL_"))]

    if not json_files:
        print("❌ 파일을 찾을 수 없습니다. 경로를 다시 확인해주세요.")
        return pd.DataFrame()

    print(f"✅ 총 {len(json_files)}개의 파일을 병합합니다.")

    for file_path in tqdm(json_files, desc="Merging Data"):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                content = json.load(f)
                
                # 디버깅 결과에 따른 실제 키 명칭 적용
                dev = content.get('deviceData', {}) # deviceDataObject 아님
                err = content.get('errorData', {})  # errorDataObject 아님
                
                # 위치 정보가 standardLocationData 안에 있는지 한 번 더 확인하며 추출
                loc = dev.get('standardLocationData', {})
                
                all_data.append({
                    'deviceId': content.get('deviceId'),
                    'deviceType': content.get('deviceType'),
                    'batteryLevel': dev.get('batteryLevel'),
                    'speed': loc.get('speed'),
                    'x': loc.get('x'),
                    'y': loc.get('y'),
                    'collision': dev.get('collision'),
                    'obstacle': dev.get('obstacle'),
                    'errorCode': err.get('errorCode'),
                    'errorState': err.get('errorState')
                })
            except:
                continue
                
    return pd.DataFrame(all_data)

if __name__ == "__main__":
    df = merge_robot_data()
    if not df.empty:
        # RTX 5060 Ti 사양 최적화: 빠른 처리를 위해 pyarrow 사용
        output_file = "robot_total_data.parquet"
        df.to_parquet(output_file, engine='pyarrow', index=False)
        print("-" * 30)
        print(f"🎉 드디어 병합 성공! 생성된 파일: {output_file}")
        print(f"데이터 개수: {len(df)}건")
        
        # 다시 한번 결측치 확인 (이번에는 0이 나와야 합니다!)
        print("\n🔍 결측치 체크 (0에 가까워야 함):")
        print(df.isnull().sum())