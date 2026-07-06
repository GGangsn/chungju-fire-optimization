import json
import urllib.request
from pathlib import Path

def main():
    print("충청북도 GeoJSON 데이터 다운로드 중...")
    url = "https://raw.githubusercontent.com/raqoon886/Local_HangJeongDong/master/hangjeongdong_%EC%B6%A9%EC%B2%AD%EB%B6%81%EB%8F%84.geojson"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            geojson_data = json.loads(response.read().decode('utf-8'))
    except Exception as e:
        print(f"다운로드 실패: {e}")
        return

    print("충주시(시군구코드: 43130) 데이터 필터링 중...")
    chungju_features = []
    
    for feature in geojson_data.get("features", []):
        props = feature.get("properties", {})
        # sgg 코드가 '43130'인 피처 필터링
        if props.get("sgg") == "43130":
            # 행정구역 이름 추출 (예: '충청북도 충주시 주덕읍' -> '주덕읍')
            full_name = props.get("adm_nm", "")
            parts = full_name.split()
            if parts:
                emd_name = parts[-1]
            else:
                emd_name = ""
            
            # 매칭을 위해 name 프로퍼티 추가
            feature["properties"]["name"] = emd_name
            chungju_features.append(feature)
            
    print(f"필터링 완료: 총 {len(chungju_features)}개 행정구역 발견.")
    
    # 충주시 전용 GeoJSON 구성
    chungju_geojson = {
        "type": "FeatureCollection",
        "name": "chungju_emd",
        "crs": geojson_data.get("crs"),
        "features": chungju_features
    }
    
    output_path = Path("chungju_emd.geojson")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chungju_geojson, f, ensure_ascii=False, indent=2)
        
    print(f"파일 저장 성공: {output_path.resolve()}")

if __name__ == "__main__":
    main()
