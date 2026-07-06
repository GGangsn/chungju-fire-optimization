import argparse
import csv
import json
import math
from pathlib import Path

REGION_COORDS = {
    "주덕읍": [37.0268, 127.7997],
    "살미면": [36.9115, 127.9734],
    "수안보면": [36.8458, 127.9822],
    "대소원면": [36.9944, 127.8183],
    "신니면": [37.0142, 127.7126],
    "노은면": [37.0726, 127.7562],
    "앙성면": [37.1394, 127.7712],
    "중앙탑면": [37.0396, 127.8647],
    "금가면": [37.0315, 127.9405],
    "동량면": [37.0253, 128.0131],
    "산척면": [37.0544, 128.0265],
    "엄정면": [37.1009, 127.9254],
    "소태면": [37.1479, 127.9482],
    "성내충인동": [36.9723, 127.9317],
    "교현안림동": [36.9744, 127.9542],
    "교현2동": [36.9806, 127.9402],
    "용산동": [36.9621, 127.9423],
    "지현동": [36.9654, 127.9298],
    "문화동": [36.9669, 127.9212],
    "호암직동": [36.9452, 127.9354],
    "달천동": [36.9533, 127.8931],
    "봉방동": [36.9815, 127.9157],
    "칠금금릉동": [36.9934, 127.9189],
    "연수동": [36.9897, 127.9463],
    "목행용탄동": [37.0125, 127.9367],
}


def clean_name(name):
    return name.replace("·", "").replace(" ", "").replace("-", "").strip()


def parse_number(value):
    return float(str(value).replace(",", "").strip())


def read_csv_rows(path):
    last_error = None
    for encoding in ("utf-8-sig", "cp949", "utf-8"):
        try:
            with open(path, "r", encoding=encoding, newline="") as file:
                return list(csv.DictReader(file))
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"CSV 인코딩을 읽지 못했습니다: {path}") from last_error


def prepare_rows(raw_rows):
    required = {"구분", "내국인(계)", "65세이상(내국)계"}
    if not raw_rows:
        raise ValueError("CSV에 데이터 행이 없습니다.")

    missing = required - set(raw_rows[0].keys())
    if missing:
        raise KeyError(f"필수 컬럼이 없습니다: {', '.join(sorted(missing))}")

    # 좌표 매핑 사전의 키들도 정제하여 검색 속도 향상
    clean_coords = {clean_name(k): v for k, v in REGION_COORDS.items()}

    rows = []
    missing_coords = []
    for raw in raw_rows:
        name = raw["구분"].strip()
        clean_key = clean_name(name)
        coords = clean_coords.get(clean_key)
        if coords is None:
            missing_coords.append(name)
            continue

        population = parse_number(raw["내국인(계)"])
        elderly = parse_number(raw["65세이상(내국)계"])
        if population <= 0:
            continue

        rows.append(
            {
                "name": name,
                "population": population,
                "elderly": elderly,
                "elderly_ratio": elderly / population * 100,
                "lat": coords[0],
                "lng": coords[1],
            }
        )

    if missing_coords:
        print(f"[경고] 좌표가 없어 제외한 지역: {', '.join(missing_coords)}")
    if len(rows) < 2:
        raise ValueError("군집화를 하려면 좌표가 있는 지역 데이터가 2개 이상 필요합니다.")
    return rows


def standardize(points):
    column_count = len(points[0])
    means = [sum(row[col] for row in points) / len(points) for col in range(column_count)]
    stds = []
    for col in range(column_count):
        variance = sum((row[col] - means[col]) ** 2 for row in points) / len(points)
        stds.append(math.sqrt(variance) or 1.0)
    return [
        [(row[col] - means[col]) / stds[col] for col in range(column_count)]
        for row in points
    ]


def distance_sq(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def kmeans(points, cluster_count, max_iter=100):
    k = min(cluster_count, len(points))
    centroids = [points[round(i * (len(points) - 1) / max(k - 1, 1))][:] for i in range(k)]
    labels = [0] * len(points)

    for _ in range(max_iter):
        changed = False
        for index, point in enumerate(points):
            label = min(range(k), key=lambda cluster: distance_sq(point, centroids[cluster]))
            if labels[index] != label:
                labels[index] = label
                changed = True

        new_centroids = []
        for cluster in range(k):
            members = [point for point, label in zip(points, labels) if label == cluster]
            if not members:
                new_centroids.append(centroids[cluster])
                continue
            new_centroids.append(
                [sum(point[col] for point in members) / len(members) for col in range(len(points[0]))]
            )

        centroids = new_centroids
        if not changed:
            break
    return labels


def minmax(values):
    low = min(values)
    high = max(values)
    if high == low:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def add_cluster_and_risk(rows, cluster_count):
    features = [[row["elderly"], row["elderly_ratio"]] for row in rows]
    labels = kmeans(standardize(features), cluster_count)

    elderly_scaled = minmax([row["elderly"] for row in rows])
    ratio_scaled = minmax([row["elderly_ratio"] for row in rows])

    for row, label, elderly_score, ratio_score in zip(rows, labels, elderly_scaled, ratio_scaled):
        row["cluster"] = label
        row["risk_score"] = (elderly_score + ratio_score) / 2

    cluster_scores = {}
    for label in sorted(set(labels)):
        members = [row for row in rows if row["cluster"] == label]
        cluster_scores[label] = sum(row["risk_score"] for row in members) / len(members)

    sorted_clusters = sorted(cluster_scores, key=cluster_scores.get, reverse=True)
    rank_by_cluster = {cluster: rank for rank, cluster in enumerate(sorted_clusters)}
    for row in rows:
        row["risk_rank"] = rank_by_cluster[row["cluster"]]
    return rows


def risk_label(rank):
    if rank == 0:
        return "고위험군"
    if rank == 1:
        return "중위험군"
    return "저위험군"


def risk_color(rank):
    if rank == 0:
        return "#d32f2f"  # 빨강
    if rank == 1:
        return "#f9a825"  # 노랑
    return "#2e7d32"      # 초록


def print_summary(rows):
    print("\n=== 군집별 평균 ===")
    clusters = sorted(set(row["cluster"] for row in rows))
    for cluster in clusters:
        members = [row for row in rows if row["cluster"] == cluster]
        avg_elderly = sum(row["elderly"] for row in members) / len(members)
        avg_ratio = sum(row["elderly_ratio"] for row in members) / len(members)
        avg_score = sum(row["risk_score"] for row in members) / len(members)
        rank = members[0]["risk_rank"]
        print(
            f"군집 {cluster} / {risk_label(rank)}: "
            f"지역 {len(members)}개, "
            f"65세 이상 평균 {avg_elderly:.1f}명, "
            f"고령인구 비율 평균 {avg_ratio:.1f}%, "
            f"위험점수 {avg_score:.3f}"
        )


def build_html(rows, geojson_data, search_radius_m):
    center_lat = sum(row["lat"] for row in rows) / len(rows)
    center_lng = sum(row["lng"] for row in rows) / len(rows)
    
    # rows 데이터를 정제된 이름을 기준으로 딕셔너리 변환
    rows_by_clean_name = {clean_name(row["name"]): row for row in rows}

    # GeoJSON 피처들에 위험 분석 정보 추가
    features_to_keep = []
    for feature in geojson_data.get("features", []):
        name = feature["properties"]["name"]
        clean_feature_name = clean_name(name)
        if clean_feature_name in rows_by_clean_name:
            row = rows_by_clean_name[clean_feature_name]
            feature["properties"]["elderly"] = int(row["elderly"])
            feature["properties"]["elderly_ratio"] = row["elderly_ratio"]
            feature["properties"]["risk_score"] = row["risk_score"]
            feature["properties"]["risk_rank"] = row["risk_rank"]
            feature["properties"]["label"] = risk_label(row["risk_rank"])
            feature["properties"]["color"] = risk_color(row["risk_rank"])
            feature["properties"]["center_lat"] = row["lat"]
            feature["properties"]["center_lng"] = row["lng"]
            features_to_keep.append(feature)

    # 매칭된 충주시 피처들만 포함하는 GeoJSON
    filtered_geojson = {
        "type": "FeatureCollection",
        "name": "chungju_emd_risk",
        "crs": geojson_data.get("crs"),
        "features": features_to_keep
    }

    # Leaflet 맵용 중심 마커 리스트 작성
    markers = [
        {
            "name": row["name"],
            "lat": row["lat"],
            "lng": row["lng"],
            "elderly": int(row["elderly"]),
            "elderly_ratio": row["elderly_ratio"],
            "risk_score": row["risk_score"],
            "label": risk_label(row["risk_rank"]),
            "color": risk_color(row["risk_rank"]),
            "radius": 6 if row["risk_rank"] == 0 else 4,
        }
        for row in rows
    ]

    markers_json = json.dumps(markers, ensure_ascii=False)
    geojson_json = json.dumps(filtered_geojson, ensure_ascii=False)

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>충주 고령자 소방 위험 군집 시뮬레이터</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/@turf/turf@6/turf.min.js"></script>
  <style>
    html, body, #map {{ width: 100%; height: 100%; margin: 0; }}
    body {{ font-family: "Malgun Gothic", "Apple SD Gothic Neo", Arial, sans-serif; }}
    .legend {{
      position: fixed;
      left: 18px;
      bottom: 18px;
      z-index: 9999;
      background: white;
      border: 1px solid #bdbdbd;
      border-radius: 6px;
      padding: 12px 14px;
      line-height: 1.6;
      font-size: 13px;
      box-shadow: 0 2px 10px rgba(0, 0, 0, 0.14);
    }}
    .legend b {{ display: block; margin-bottom: 6px; font-size: 14px; }}
    .dot {{ display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }}
    .info-box {{ margin-top: 8px; font-size: 11px; color: #616161; border-top: 1px solid #e0e0e0; padding-top: 6px; }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="legend">
    <b>충주 고령자 소방 위험 군집</b>
    <span class="dot" style="background:#d32f2f"></span>고위험군 (반투명)<br>
    <span class="dot" style="background:#f9a825"></span>중위험군 (반투명)<br>
    <span class="dot" style="background:#2e7d32"></span>저위험군 (반투명)<br>
    <span style="font-size:14px; vertical-align:middle; margin-right:4px;">🚒</span>실제 소방서 &nbsp; <span style="font-size:14px; vertical-align:middle; margin-right:4px;">🏥</span>실제 종합병원<br>
    <div class="info-box">
      • 지도/폴리곤 클릭: <b>가상 소방안전센터</b> 핀 설치<br>
      • 설치 시 주변 위험 등급 실시간 완화(안전성 시뮬레이션)<br>
      • 더블클릭/빈곳 재클릭: 필터 초기화
    </div>
  </div>
  <script>
    const map = L.map("map").setView([{center_lat:.6f}, {center_lng:.6f}], 11.4);
    
    // 어둡고 대비가 강한 스타일의 지도 타일 적용하여 고채도 포인트 컬러 및 폴리곤 시인성 극대화
    L.tileLayer("https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}{{r}}.png", {{
      maxZoom: 19,
      attribution: "&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors &copy; <a href='https://carto.com/attributions'>CARTO</a>"
    }}).addTo(map);

    const geojsonData = {geojson_json};
    const markers = {markers_json};
    const searchRadiusMeters = {int(search_radius_m)};
    
    let selectedPin = null;
    let selectedRadius = null;
    let geojsonLayer = null;

    // GeoJSON 레이어 생성 및 초기 반투명 스타일 지정
    geojsonLayer = L.geoJSON(geojsonData, {{
      style: function(feature) {{
        return {{
          fillColor: feature.properties.color,
          fillOpacity: 0.22, // 반투명 설정
          color: feature.properties.color,
          weight: 1.5,
          opacity: 0.45
        }};
      }},
      onEachFeature: function(feature, layer) {{
        layer.bindTooltip(`<b>${{feature.properties.name}}</b> (${{feature.properties.label}})`, {{
          sticky: true
        }});
        
        // 폴리곤 클릭 시 가상 119 안전센터 핀을 올바르게 설치하도록 맵 이벤트와 연동
        layer.on("click", function(e) {{
          onMapClick(e);
          L.DomEvent.stopPropagation(e); // 이벤트가 버블링되어 맵에 이중으로 찍히지 않도록 방지
        }});
      }}
    }}).addTo(map);

    // Turf.js를 사용하여 충주시 전체 외곽 경계를 검은색으로 강조
    try {{
      const unioned = geojsonData.features.reduce((prev, curr) => {{
        return turf.union(prev, curr);
      }});
      if (unioned) {{
        L.geoJSON(unioned, {{
          style: {{
            color: "#1e1e1e",      // 충주시 외곽 경계선 (검은색)
            weight: 3.5,           // 경계선 굵기
            opacity: 0.95,         // 선명하게
            fill: false,           // 읍면동 색상 유지를 위해 내부 채우기 없음
            interactive: false
          }}
        }}).addTo(map);
      }}
    }} catch (e) {{
      console.error("Turf union failed:", e);
    }}

    // 🚑 실제 소방서 및 병원 마커 추가
    const fireStationIcon = L.divIcon({{
      html: '<div style="background-color: #d32f2f; color: white; border-radius: 50%; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; font-size: 13px; border: 2.2px solid white; box-shadow: 0 2px 6px rgba(0,0,0,0.35); font-weight: bold; cursor: pointer;">🚒</div>',
      className: 'custom-div-icon',
      iconSize: [24, 24],
      iconAnchor: [12, 12]
    }});

    const hospitalIcon = L.divIcon({{
      html: '<div style="background-color: #1976d2; color: white; border-radius: 50%; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; font-size: 13px; border: 2.2px solid white; box-shadow: 0 2px 6px rgba(0,0,0,0.35); font-weight: bold; cursor: pointer;">🏥</div>',
      className: 'custom-div-icon',
      iconSize: [24, 24],
      iconAnchor: [12, 12]
    }});

    const realFireStations = [
      {{ name: "충주소방서", lat: 36.9818, lng: 127.9333, desc: "충주시 소방 총괄 본부" }},
      {{ name: "주덕119안전센터", lat: 36.9739, lng: 127.8015, desc: "주덕읍 및 인근 면 지역 대응" }},
      {{ name: "수안보119안전센터", lat: 36.8407, lng: 127.9944, desc: "수안보 및 충주 남부 산간 대응" }},
      {{ name: "앙성119안전센터", lat: 37.1471, lng: 127.7656, desc: "앙성면 및 북서부 외곽 대응" }},
      {{ name: "연수119안전센터", lat: 36.9892, lng: 127.9392, desc: "도심 고인구 밀집 구역 대응" }},
      {{ name: "중앙탑119안전센터", lat: 37.0163, lng: 127.8631, desc: "중앙탑면 및 서부 기업도시 대응" }}
    ];

    const realHospitals = [
      {{ name: "건국대학교 충주병원", lat: 36.9767, lng: 127.9272, desc: "종합병원 (응급의료기관)" }},
      {{ name: "충청북도 충주의료원", lat: 36.9634, lng: 127.9620, desc: "지역 거점 공공병원" }},
      {{ name: "세명대학교 한방병원", lat: 36.9678, lng: 127.9295, desc: "한방 특화 병원" }}
    ];

    realFireStations.forEach((st) => {{
      L.marker([st.lat, st.lng], {{ icon: fireStationIcon }}).addTo(map)
        .bindPopup(`<b>🔥 ${{st.name}}</b><br>${{st.desc}}`);
    }});

    realHospitals.forEach((hp) => {{
      L.marker([hp.lat, hp.lng], {{ icon: hospitalIcon }}).addTo(map)
        .bindPopup(`<b>🏥 ${{hp.name}}</b><br>${{hp.desc}}`);
    }});

    // 각 지역 중심점에 원형 마커 표시 (중심 좌표 시각화)
    markers.forEach((area) => {{
      L.circleMarker([area.lat, area.lng], {{
        radius: area.radius,
        color: "#ffffff",
        weight: 1.2,
        fill: true,
        fillColor: area.color,
        fillOpacity: 0.9,
        interactive: false
      }}).addTo(map);
    }});

    // 가상 안전센터 거리별 안전도 완화 색상 계산 함수
    function getSafetyColor(originalRank, distance) {{
      let currentRank = originalRank; // 0:고위험, 1:중위험, 2:저위험
      
      if (distance <= 2000) {{
        // 반경 2.0km 이내 (최고 혜택): 위험도를 2단계 낮춤 -> 무조건 저위험(안전 - 초록)으로 변경
        currentRank = 2;
      }} else if (distance <= 4000) {{
        // 반경 2.0km 초과 ~ 4.0km 이하 (보통 혜택): 위험도를 1단계 완화
        currentRank = Math.min(2, originalRank + 1); // 고(0)->중(1), 중(1)->저(2), 저(2)->저(2)
      }}
      
      // 색상 코드 반환
      if (currentRank === 0) return "#d32f2f"; // 빨강
      if (currentRank === 1) return "#f9a825"; // 노랑
      return "#2e7d32";                         // 초록
    }}

    function clearRiskSelection() {{
      if (selectedPin) map.removeLayer(selectedPin);
      if (selectedRadius) map.removeLayer(selectedRadius);
      selectedPin = null;
      selectedRadius = null;

      // 모든 폴리곤의 스타일을 원래의 반투명 상태로 복구
      geojsonLayer.eachLayer((layer) => {{
        const color = layer.feature.properties.color;
        layer.setStyle({{
          fillColor: color,
          fillOpacity: 0.22,
          color: color,
          weight: 1.5,
          opacity: 0.45
        }});
      }});
    }}

    function onMapClick(e) {{
      if (selectedPin && e.latlng.distanceTo(selectedPin.getLatLng()) < 500) {{
        clearRiskSelection();
        return;
      }}
      
      clearRiskSelection();

      // 파란색 핀 = 가상 119 안전센터 설치
      selectedPin = L.marker(e.latlng).addTo(map);
      
      // 검색 반경 2.0km를 연한 빨간색 점선으로 강조
      selectedRadius = L.circle(e.latlng, {{
        radius: searchRadiusMeters,
        color: "#d32f2f",
        weight: 1.2,
        dashArray: "4, 4",
        fillColor: "#ef5350",
        fillOpacity: 0.03,
        interactive: false
      }}).addTo(map);

      const includedHighRisk = [];
      const includedAll = [];
      const alleviatedList = []; // 안전도가 완화된 구역 리스트

      // 실시간으로 각 행정구역 폴리곤 스타일 변경
      geojsonLayer.eachLayer((layer) => {{
        const props = layer.feature.properties;
        const centerLatLng = L.latLng(props.center_lat, props.center_lng);
        const distance = e.latlng.distanceTo(centerLatLng);

        // 가상 안전센터와의 거리에 따른 새로운 등급 색상 적용
        const newColor = getSafetyColor(props.risk_rank, distance);
        const isAlleviated = (props.risk_rank === 0 && newColor !== "#d32f2f") || (props.risk_rank === 1 && newColor === "#2e7d32");

        if (distance <= searchRadiusMeters) {{
          // 반경 내 지역: 고채도 강조 렌더링
          layer.setStyle({{
            fillColor: newColor,
            fillOpacity: 0.65,
            color: newColor,
            weight: 3.5,
            opacity: 0.95
          }});
          
          includedAll.push(props.name);
          if (props.risk_rank === 0) {{
            includedHighRisk.push(props.name);
          }}
          if (isAlleviated) {{
            alleviatedList.push(`${{props.name}}`);
          }}
        }} else {{
          // 반경 외 지역: 2~4km 범위면 색상 변경은 반영하되 약간 반투명하게, 4km 초과는 거의 투명하게 처리
          const isSubSafe = distance <= 4000;
          layer.setStyle({{
            fillColor: newColor,
            fillOpacity: isSubSafe ? 0.20 : 0.04,
            color: newColor,
            weight: isSubSafe ? 1.5 : 0.6,
            opacity: isSubSafe ? 0.4 : 0.12
          }});
          
          if (isSubSafe && isAlleviated) {{
            alleviatedList.push(`${{props.name}}`);
          }}
        }}
      }});

      // 핀에 팝업 정보 바인딩 및 오픈
      let popupContent = `<b>🚒 가상 119안전센터 설립 후보지</b><br><br>`;
      if (includedAll.length > 0) {{
        popupContent += `• <b>2.0km 내 대응 구역:</b> ${{includedAll.join(", ")}}<br>`;
      }} else {{
        popupContent += `• 2km 내 인접 행정구역 중심점 없음.<br>`;
      }}

      if (alleviatedList.length > 0) {{
        popupContent += `<span style="color:#2e7d32;">• <b>안전 등급 개선 지역:</b> ${{alleviatedList.join(", ")}}</span><br>`;
      }}
      
      popupContent += `<br><span style="color:#1565c0;">• 핀 주변 지역의 소방 도착 골든타임이 단축되어 위험도가 완화되었습니다!</span>`;

      selectedPin.bindPopup(popupContent).openPopup();
    }}

    map.on("click", onMapClick);
    
    // 지도를 더블클릭하면 핀 선택을 해제하고 초기 상태로 리셋
    map.on("dblclick", (e) => {{
      L.DomEvent.stopPropagation(e);
      clearRiskSelection();
    }});
  </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="충주시 고령자 소방 위험 군집 지도 생성기 (GeoJSON 연동)")
    parser.add_argument("--csv", default="충청북도 충주시_인구통계_20251231.csv")
    parser.add_argument("--geojson", default="chungju_emd.geojson")
    parser.add_argument("--output", default="chungju_fire_cluster_simulator.html")
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--search-radius-m", type=int, default=2000)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    geojson_path = Path(args.geojson)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path.resolve()}")
    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON 파일을 찾을 수 없습니다: {geojson_path.resolve()}")

    # 1. 인구 통계 데이터 로드 및 전처리
    raw_rows = read_csv_rows(csv_path)
    rows = prepare_rows(raw_rows)

    # 2. K-Means 알고리즘 및 위험 지수 평가 적용
    rows = add_cluster_and_risk(rows, args.clusters)
    print_summary(rows)

    # 3. GeoJSON 데이터 불러오기
    with open(geojson_path, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)

    # 4. 지도 시각화 HTML 생성
    html_content = build_html(rows, geojson_data, args.search_radius_m)
    
    output_path = Path(args.output)
    output_path.write_text(html_content, encoding="utf-8")
    print(f"\n성공적으로 지도를 빌드했습니다: {output_path.resolve()}")


if __name__ == "__main__":
    main()
