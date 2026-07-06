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

    rows = []
    missing_coords = []
    for raw in raw_rows:
        name = raw["구분"].strip()
        coords = REGION_COORDS.get(name)
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
    
    # rows 데이터를 name 기준으로 조회하기 쉽게 딕셔너리로 변환
    rows_by_name = {row["name"]: row for row in rows}

    # GeoJSON 피처들에 위험 분석 정보 추가
    features_to_keep = []
    for feature in geojson_data.get("features", []):
        name = feature["properties"]["name"]
        if name in rows_by_name:
            row = rows_by_name[name]
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
    html, body, #map { width: 100%; height: 100%; margin: 0; }
    body { font-family: "Malgun Gothic", "Apple SD Gothic Neo", Arial, sans-serif; }
    .legend {
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
    }
    .legend b { display: block; margin-bottom: 6px; font-size: 14px; }
    .dot { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }
    .info-box { margin-top: 8px; font-size: 11px; color: #616161; border-top: 1px solid #e0e0e0; padding-top: 6px; }
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="legend">
    <b>충주 고령자 소방 위험 군집</b>
    <span class="dot" style="background:#d32f2f"></span>고위험군 (반투명)<br>
    <span class="dot" style="background:#f9a825"></span>중위험군 (반투명)<br>
    <span class="dot" style="background:#2e7d32"></span>저위험군 (반투명)<br>
    <div class="info-box">
      • 지도 클릭: 주변 2km 내 지역 강조 (실시간 변경)<br>
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
        layer.bindPopup(
          `<b>${{feature.properties.name}}</b><br>` +
          `등급: ${{feature.properties.label}}<br>` +
          `65세 이상: ${{feature.properties.elderly.toLocaleString()}}명<br>` +
          `고령인구 비율: ${{feature.properties.elderly_ratio.toFixed(1)}}%<br>` +
          `위험점수: ${{feature.properties.risk_score.toFixed(3)}}`
        );
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
      // 이미 핀이 꽂혀 있는 경우, 지도의 다른 지점을 클릭하면 지우고 새로 생성 (더블클릭처럼 사용 가능)
      if (selectedPin && e.latlng.distanceTo(selectedPin.getLatLng()) < 500) {{
        clearRiskSelection();
        return;
      }}
      
      clearRiskSelection();

      selectedPin = L.marker(e.latlng).addTo(map);
      
      // 검색 반경 2.0km를 연한 빨간색 점선으로 강조
      selectedRadius = L.circle(e.latlng, {{
        radius: searchRadiusMeters,
        color: "#d32f2f",
        weight: 1.2,
        dashArray: "4, 4",
        fillColor: "#ef5350",
        fillOpacity: 0.04,
        interactive: false
      }}).addTo(map);

      const includedHighRisk = [];
      const includedAll = [];

      // 실시간으로 각 행정구역 폴리곤 스타일 변경
      geojsonLayer.eachLayer((layer) => {{
        const props = layer.feature.properties;
        const centerLatLng = L.latLng(props.center_lat, props.center_lng);
        const distance = e.latlng.distanceTo(centerLatLng);

        if (distance <= searchRadiusMeters) {{
          // 반경 내 지역: 위험군별 색상을 아주 선명하고 두껍게 강조
          layer.setStyle({{
            fillOpacity: 0.60,
            weight: 3.0,
            opacity: 0.9
          }});
          
          includedAll.push(props.name);
          if (props.risk_rank === 0) {{
            includedHighRisk.push(props.name);
          }}
        }} else {{
          // 반경 외 지역: 페이드아웃 (투명하고 연하게 처리하여 대비 효과 극대화)
          layer.setStyle({{
            fillOpacity: 0.04,
            weight: 0.6,
            opacity: 0.12
          }});
        }}
      }});

      // 핀에 팝업 정보 바인딩 및 오픈
      let popupContent = `<b>선택 지점</b><br>반경 2.0km 내 분석 결과:<br>`;
      if (includedAll.length > 0) {{
        popupContent += `• 대상 지역: ${{includedAll.join(", ")}}<br>`;
        if (includedHighRisk.length > 0) {{
          popupContent += `<span style="color:#d32f2f;">• <b>고위험군 포함</b>: ${{includedHighRisk.join(", ")}}</span>`;
        }} else {{
          popupContent += `<span style="color:#2e7d32;">• 고위험군 없음</span>`;
        }}
      }} else {{
        popupContent += `반경 내 포함되는 행정구역 중심점 없음.`;
      }}

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
