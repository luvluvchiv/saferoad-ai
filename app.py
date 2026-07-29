
from pathlib import Path

import json

import folium
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier, Pool
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium


# =========================
# ตั้งค่าหน้าเว็บ
# =========================
st.set_page_config(
    page_title="SafeRoad AI",
    layout="wide",
)


# =========================
# ตำแหน่งไฟล์
# =========================
BASE_DIR = Path(__file__).resolve().parent

SCORED_PATH = (
    BASE_DIR
    / "outputs/test_2025_fatal_risk_scored.csv"
)

GRID_PATH = (
    BASE_DIR
    / "outputs/risk_priority_grids_2025.csv"
)

TOP_URGENT_PATH = (
    BASE_DIR
    / "outputs/top_20_urgent_areas_dashboard_2025.csv"
)

PROVINCE_PATH = (
    BASE_DIR
    / "outputs/province_priority_summary_2025.csv"
)


MODEL_PATH = (
    BASE_DIR
    / "models/saferoad_fatal_risk_final.cbm"
)

METADATA_PATH = (
    BASE_DIR
    / "models/saferoad_fatal_risk_final_metadata.json"
)


# =========================
# โหลดข้อมูล
# =========================
@st.cache_data
def load_data():
    accidents = pd.read_csv(
        SCORED_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )

    grids = pd.read_csv(
        GRID_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )

    urgent_areas = pd.read_csv(
        TOP_URGENT_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )

    provinces = pd.read_csv(
        PROVINCE_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )

    accidents["time"] = pd.to_datetime(
        accidents["time"],
        errors="coerce",
    )

    return accidents, grids, urgent_areas, provinces


try:
    test_2025, reliable_grids, top_urgent, provinces = load_data()

except FileNotFoundError as error:
    st.error(f"ไม่พบไฟล์ที่จำเป็น: {error}")
    st.stop()


@st.cache_resource
def load_prediction_model():
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        model_metadata = json.load(file)

    return model, model_metadata


try:
    prediction_model, model_metadata = (
        load_prediction_model()
    )

except FileNotFoundError as error:
    st.error(f"ไม่พบไฟล์โมเดล: {error}")
    st.stop()


final_features = model_metadata["features"]

final_categorical_features = model_metadata[
    "categorical_features"
]

screening_threshold = float(
    model_metadata.get(
        "screening_threshold",
        0.40,
    )
)

critical_threshold = float(
    model_metadata.get(
        "critical_threshold",
        0.65,
    )
)


# ถ้าไฟล์กริดยังไม่มี priority_level ให้คำนวณใหม่
if "priority_level" not in reliable_grids.columns:
    watch_threshold = reliable_grids[
        "priority_score"
    ].quantile(0.60)

    urgent_threshold = reliable_grids[
        "priority_score"
    ].quantile(0.90)

    reliable_grids["priority_level"] = np.select(
        [
            reliable_grids["priority_score"]
            >= urgent_threshold,

            reliable_grids["priority_score"]
            >= watch_threshold,
        ],
        [
            "urgent",
            "watch",
        ],
        default="general",
    )


def format_hour(value):
    if pd.isna(value):
        return "ไม่ระบุ"

    try:
        return f"{int(float(value)):02d}:00 น."
    except (TypeError, ValueError):
        return str(value)



# =========================
# Sidebar
# =========================
st.sidebar.title("SafeRoad AI")

page = st.sidebar.radio(
    "เลือกหน้าที่ต้องการดู",
    [
        "ภาพรวม",
        "แผนที่พื้นที่เร่งด่วน",
        "อันดับจังหวัด",
        "ประเมินความเสี่ยง",
    ],
)

st.sidebar.divider()

st.sidebar.caption(
    "ข้อมูลอุบัติเหตุทางถนนปี 2025"
)

st.sidebar.warning(
    "Fatal Risk Score เป็นคะแนนสำหรับคัดกรองและ"
    "จัดลำดับความสำคัญ ไม่ใช่การยืนยันว่าจะมีผู้เสียชีวิต"
)


# =========================
# ส่วนหัว
# =========================
st.title("SafeRoad AI")
st.subheader(
    "ระบบคัดกรองและจัดลำดับพื้นที่เสี่ยงจากอุบัติเหตุรุนแรง"
)


# =========================
# หน้า 1: ภาพรวม
# =========================
if page == "ภาพรวม":

    st.markdown(
        "วิเคราะห์ข้อมูลอุบัติเหตุปี 2025 "
        "เพื่อช่วยระบุพื้นที่และรูปแบบเหตุการณ์ที่ควรเฝ้าระวัง"
    )

    total_accidents = len(test_2025)

    critical_count = int(
        (test_2025["risk_level"] == "critical").sum()
    )

    high_count = int(
        (test_2025["risk_level"] == "high").sum()
    )

    urgent_grid_count = int(
        (
            reliable_grids["priority_level"]
            == "urgent"
        ).sum()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "อุบัติเหตุทั้งหมด",
        f"{total_accidents:,}",
    )

    col2.metric(
        "High Risk",
        f"{high_count:,}",
    )

    col3.metric(
        "Critical Risk",
        f"{critical_count:,}",
    )

    col4.metric(
        "พื้นที่เร่งด่วน",
        f"{urgent_grid_count:,}",
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("จำนวนเหตุการณ์ตามระดับความเสี่ยง")

        risk_order = ["low", "high", "critical"]

        risk_distribution = (
            test_2025["risk_level"]
            .value_counts()
            .reindex(risk_order, fill_value=0)
            .rename_axis("ระดับความเสี่ยง")
            .to_frame("จำนวนเหตุการณ์")
        )

        st.bar_chart(risk_distribution)

    with right:
        st.subheader("จังหวัดที่ควรตรวจสอบก่อน")

        province_display = (
            provinces
            .sort_values(
                "province_priority_score",
                ascending=False,
            )
            .head(10)
            .copy()
        )

        province_display.insert(
            0,
            "อันดับ",
            range(1, len(province_display) + 1),
        )

        overview_columns = [
            "อันดับ",
            "province",
            "province_priority_score",
            "accident_count",
            "critical_count",
        ]

        st.dataframe(
            province_display[overview_columns].rename(
                columns={
                    "province": "จังหวัด",
                    "province_priority_score": "Priority Score",
                    "accident_count": "จำนวนเหตุการณ์",
                    "critical_count": "Critical Events",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.info(
        "พื้นที่ที่มีคะแนนสูง หมายถึงพื้นที่ที่ระบบเสนอให้"
        "ตรวจสอบก่อนจากรูปแบบความเสี่ยงและจำนวนเหตุการณ์"
        "ในชุดข้อมูล ไม่ได้หมายถึงพื้นที่อันตรายที่สุดของประเทศไทย"
    )


# =========================
# หน้า 2: แผนที่
# =========================
elif page == "แผนที่พื้นที่เร่งด่วน":

    st.markdown(
        "Heatmap แสดงภาพรวม Priority Score "
        "ส่วนหมุดแสดงพื้นที่เร่งด่วน 10% แรก"
    )

    risk_map = folium.Map(
        location=[13.5, 101.0],
        zoom_start=6,
        tiles="CartoDB positron",
    )

    heat_data = (
        reliable_grids[
            [
                "grid_lat",
                "grid_lon",
                "priority_score",
            ]
        ]
        .dropna()
        .values
        .tolist()
    )

    HeatMap(
        heat_data,
        name="ภาพรวม Priority Score",
        radius=14,
        blur=18,
        min_opacity=0.30,
        max_zoom=10,
    ).add_to(risk_map)

    urgent_cluster = MarkerCluster(
        name="พื้นที่เร่งด่วน"
    ).add_to(risk_map)

    urgent_grids = reliable_grids[
        reliable_grids["priority_level"] == "urgent"
    ]

    for _, row in urgent_grids.iterrows():

        province_name = row.get(
            "main_province",
            "ไม่ระบุ",
        )

        vehicle = row.get(
            "main_vehicle",
            "ไม่ระบุ",
        )

        road_type = row.get(
            "main_road_type",
            "ไม่ระบุ",
        )

        peak_hour = row.get(
            "peak_hour",
            np.nan,
        )

        if pd.isna(peak_hour):
            peak_hour_text = "ไม่ระบุ"
        else:
            peak_hour_text = f"{int(peak_hour):02d}:00 น."

        popup_text = f"""
        <div style="width:280px">
            <h4>SafeRoad AI</h4>
            <b>จังหวัดหลัก:</b> {province_name}<br>
            <b>Priority Score:</b>
            {row['priority_score']:.2f}/100<br>
            <b>จำนวนเหตุการณ์:</b>
            {int(row['accident_count'])}<br>
            <b>Fatal Risk เฉลี่ย:</b>
            {row['mean_risk_score']:.3f}<br>
            <b>Critical Events:</b>
            {int(row['critical_count'])}<br><br>
            <b>กลุ่มหลัก:</b> {vehicle}<br>
            <b>ประเภทถนนหลัก:</b> {road_type}<br>
            <b>ช่วงเวลาที่พบบ่อย:</b>
            {peak_hour_text}
        </div>
        """

        folium.CircleMarker(
            location=[
                row["grid_lat"],
                row["grid_lon"],
            ],
            radius=min(
                5 + np.sqrt(row["accident_count"]),
                15,
            ),
            color="red",
            weight=2,
            fill=True,
            fill_color="red",
            fill_opacity=0.75,
            popup=folium.Popup(
                popup_text,
                max_width=350,
            ),
        ).add_to(urgent_cluster)

    folium.LayerControl(
        collapsed=False
    ).add_to(risk_map)

    st_folium(
        risk_map,
        width=None,
        height=650,
        returned_objects=[],
    )

    st.caption(
        "Heatmap แสดงการกระจุกตัวของ Priority Score "
        "และหมุดแดงแสดงกริดที่อยู่ในกลุ่มเร่งด่วน 10% แรก"
    )

    st.subheader("Top 20 พื้นที่เร่งด่วน")

    urgent_display = top_urgent.copy()

    urgent_display = urgent_display.sort_values(
        "priority_score",
        ascending=False,
    ).head(20)

    if "peak_hour" in urgent_display.columns:
        urgent_display["peak_hour"] = (
            urgent_display["peak_hour"]
            .apply(format_hour)
        )

    urgent_display.insert(
        0,
        "อันดับ",
        range(1, len(urgent_display) + 1),
    )

    urgent_columns = [
        "อันดับ",
        "main_province",
        "priority_score",
        "accident_count",
        "mean_risk_score",
        "critical_count",
        "main_vehicle",
        "main_road_type",
        "peak_hour",
    ]

    available_urgent_columns = [
        column
        for column in urgent_columns
        if column in urgent_display.columns
    ]

    st.dataframe(
        urgent_display[
            available_urgent_columns
        ].rename(
            columns={
                "main_province": "จังหวัดหลัก",
                "priority_score": "Priority Score",
                "accident_count": "จำนวนเหตุการณ์",
                "mean_risk_score": "Fatal Risk เฉลี่ย",
                "critical_count": "Critical Events",
                "main_vehicle": "กลุ่มหลัก",
                "main_road_type": "ประเภทถนน",
                "peak_hour": "ช่วงเวลาหลัก",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )


# =========================
# หน้า 3: อันดับจังหวัด
# =========================
elif page == "อันดับจังหวัด":

    st.markdown(
        "การจัดอันดับระดับจังหวัดใช้คะแนนความเสี่ยง "
        "ปริมาณเหตุการณ์ และสัดส่วน Critical Events"
    )

    province_table = provinces.sort_values(
        "province_priority_score",
        ascending=False,
    ).copy()

    province_table.insert(
        0,
        "อันดับ",
        range(1, len(province_table) + 1),
    )
    if "high_risk_peak_hour" in province_table.columns:
        province_table["high_risk_peak_hour"] = (
            province_table["high_risk_peak_hour"]
            .apply(format_hour)
        )


    province_columns = [
        "อันดับ",
        "province",
        "province_priority_score",
        "accident_count",
        "smoothed_mean_risk",
        "critical_count",
        "smoothed_critical_rate",
        "high_risk_vehicle",
        "high_risk_road_type",
        "high_risk_peak_hour",
        "evidence_level",
    ]

    available_province_columns = [
        column
        for column in province_columns
        if column in province_table.columns
    ]

    st.dataframe(
        province_table[
            available_province_columns
        ].rename(
            columns={
                "province": "จังหวัด",
                "province_priority_score": "Priority Score",
                "accident_count": "จำนวนเหตุการณ์",
                "smoothed_mean_risk": "Fatal Risk เฉลี่ย",
                "critical_count": "Critical Events",
                "smoothed_critical_rate": "Critical Rate",
                "high_risk_vehicle": "กลุ่มหลัก",
                "high_risk_road_type": "ประเภทถนนหลัก",
                "high_risk_peak_hour": "ช่วงเวลาหลัก",
                "evidence_level": "ระดับข้อมูล",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "มีการปรับค่าแบบ Smoothing และตัดจังหวัดที่มีข้อมูล"
        "น้อยกว่า 50 เหตุการณ์ออก เพื่อลดความผันผวนของอันดับ"
    )


# =========================
# หน้า 4: ประเมินความเสี่ยง
# =========================
elif page == "ประเมินความเสี่ยง":

    st.markdown(
        "กรอกข้อมูลลักษณะเหตุการณ์เพื่อให้โมเดล"
        "ประเมิน Fatal Risk Score"
    )

    st.caption(
        "ระบบใช้พิกัดตัวแทนของจังหวัดจากค่ากึ่งกลาง"
        "ของข้อมูลปี 2025"
    )

    def get_options(
        column,
        remove_unspecified=False,
    ):
        values = (
            test_2025[column]
            .dropna()
            .astype(str)
            .str.strip()
        )

        values = values[values != ""]

        if remove_unspecified:
            values = values[
                values != "ไม่ระบุ"
            ]

        return sorted(values.unique().tolist())


    province_options = get_options(
        "province",
        remove_unspecified=True,
    )

    vehicle_options = get_options(
        "first_vehicle"
    )

    road_options = get_options(
        "road_type"
    )

    terrain_options = get_options(
        "terrain_type"
    )

    weather_options = get_options(
        "weather"
    )


    with st.form("fatal_risk_form"):

        left, right = st.columns(2)

        with left:
            selected_province = st.selectbox(
                "จังหวัด",
                province_options,
            )

            selected_vehicle = st.selectbox(
                "ประเภทผู้ใช้ถนนหรือยานพาหนะ",
                vehicle_options,
            )

            selected_road = st.selectbox(
                "ประเภทถนน",
                road_options,
            )

            selected_terrain = st.selectbox(
                "ลักษณะพื้นที่",
                terrain_options,
            )

        with right:
            selected_weather = st.selectbox(
                "สภาพอากาศ",
                weather_options,
            )

            selected_date = st.date_input(
                "วันที่",
                value=pd.Timestamp(
                    "2025-01-01"
                ).date(),
            )

            selected_hour = st.slider(
                "เวลาที่เกิดเหตุ",
                min_value=0,
                max_value=23,
                value=18,
                step=1,
                format="%02d:00 น.",
            )

            st.caption(
                f"เวลาที่เลือก: "
                f"{selected_hour:02d}:00 น."
            )

        submitted = st.form_submit_button(
            "ประเมินความเสี่ยง",
            use_container_width=True,
        )


    if submitted:

        province_rows = test_2025[
            test_2025["province"]
            == selected_province
        ]

        representative_latitude = pd.to_numeric(
            province_rows["latitude"],
            errors="coerce",
        ).median()

        representative_longitude = pd.to_numeric(
            province_rows["longitude"],
            errors="coerce",
        ).median()

        if pd.isna(representative_latitude):
            representative_latitude = pd.to_numeric(
                test_2025["latitude"],
                errors="coerce",
            ).median()

        if pd.isna(representative_longitude):
            representative_longitude = pd.to_numeric(
                test_2025["longitude"],
                errors="coerce",
            ).median()

        selected_month = selected_date.month

        selected_day_of_week = (
            selected_date.weekday()
        )

        selected_is_weekend = int(
            selected_day_of_week >= 5
        )

        input_values = {
            "province": selected_province,
            "first_vehicle": selected_vehicle,
            "road_type": selected_road,
            "terrain_type": selected_terrain,
            "weather": selected_weather,
            "hour": selected_hour,
            "month": selected_month,
            "day_of_week": selected_day_of_week,
            "is_weekend": selected_is_weekend,
            "latitude": float(
                representative_latitude
            ),
            "longitude": float(
                representative_longitude
            ),
        }

        input_row = pd.DataFrame(
            [
                {
                    feature: input_values[feature]
                    for feature in final_features
                }
            ]
        )

        for column in final_categorical_features:
            input_row[column] = (
                input_row[column]
                .fillna("ไม่ระบุ")
                .astype(str)
            )

        fatal_risk_score = float(
            prediction_model.predict_proba(
                input_row
            )[0, 1]
        )

        if fatal_risk_score >= critical_threshold:
            risk_level = "Critical"
            suggested_action = (
                "ควรจัดลำดับตรวจสอบพื้นที่ก่อน "
                "และพิจารณามาตรการด้านความเร็ว "
                "ทัศนวิสัย และความพร้อมฉุกเฉิน"
            )

        elif fatal_risk_score >= screening_threshold:
            risk_level = "High"
            suggested_action = (
                "ควรเฝ้าระวังและตรวจสอบปัจจัย"
                "ด้านถนน ช่วงเวลา และผู้ใช้ถนน"
            )

        else:
            risk_level = "Low"
            suggested_action = (
                "อยู่ในระดับเฝ้าระวังทั่วไป "
                "แต่ยังควรใช้มาตรการความปลอดภัย"
                "ตามปกติ"
            )

        st.divider()
        st.subheader("ผลการประเมิน")

        result_col1, result_col2 = st.columns(2)

        result_col1.metric(
            "Fatal Risk Score",
            f"{fatal_risk_score * 100:.1f}/100",
        )

        result_col2.metric(
            "ระดับความเสี่ยง",
            risk_level,
        )

        st.progress(
            min(
                max(fatal_risk_score, 0.0),
                1.0,
            )
        )

        st.markdown(
            f"**แนวทางเบื้องต้น:** "
            f"{suggested_action}"
        )

        st.caption(
            "Fatal Risk Score เป็นคะแนนคัดกรอง"
            "จากรูปแบบข้อมูลย้อนหลัง ไม่ใช่"
            "ความน่าจะเป็นที่ยืนยันว่าจะมีผู้เสียชีวิต"
        )

        st.subheader("ข้อมูลที่ใช้ประเมิน")

        input_summary = pd.DataFrame(
            {
                "รายการ": [
                    "จังหวัด",
                    "ประเภทผู้ใช้ถนนหรือยานพาหนะ",
                    "ประเภทถนน",
                    "ลักษณะพื้นที่",
                    "สภาพอากาศ",
                    "วันที่",
                    "เวลาที่เกิดเหตุ",
                    "พิกัดตัวแทน",
                ],
                "ค่าที่เลือก": [
                    selected_province,
                    selected_vehicle,
                    selected_road,
                    selected_terrain,
                    selected_weather,
                    selected_date.strftime(
                        "%d/%m/%Y"
                    ),
                    f"{selected_hour:02d}:00 น.",
                    (
                        f"{representative_latitude:.4f}, "
                        f"{representative_longitude:.4f}"
                    ),
                ],
            }
        )

        st.dataframe(
            input_summary,
            hide_index=True,
            use_container_width=True,
        )


        st.subheader("ปัจจัยที่มีผลต่อคะแนน")

        try:
            prediction_pool = Pool(
                input_row,
                cat_features=(
                    final_categorical_features
                ),
            )

            local_shap = (
                prediction_model
                .get_feature_importance(
                    prediction_pool,
                    type="ShapValues",
                )[0][:-1]
            )

            feature_labels = {
                "province": "จังหวัด",
                "first_vehicle": (
                    "ประเภทผู้ใช้ถนนหรือยานพาหนะ"
                ),
                "road_type": "ประเภทถนน",
                "terrain_type": "ลักษณะพื้นที่",
                "weather": "สภาพอากาศ",
                "hour": "เวลาที่เกิดเหตุ",
                "month": "เดือน",
                "day_of_week": "วันในสัปดาห์",
                "is_weekend": "วันหยุดสุดสัปดาห์",
                "latitude": "ละติจูดตัวแทน",
                "longitude": "ลองจิจูดตัวแทน",
            }

            explanation_table = pd.DataFrame(
                {
                    "feature": final_features,
                    "ค่าที่ใช้": [
                        input_row.iloc[0][feature]
                        for feature in final_features
                    ],
                    "shap_value": local_shap,
                }
            )

            explanation_table["ความสำคัญ"] = (
                explanation_table[
                    "shap_value"
                ].abs()
            )

            explanation_table["ทิศทาง"] = np.select(
                [
                    explanation_table[
                        "shap_value"
                    ] > 0,

                    explanation_table[
                        "shap_value"
                    ] < 0,
                ],
                [
                    "เพิ่มคะแนนความเสี่ยง",
                    "ลดคะแนนความเสี่ยง",
                ],
                default="แทบไม่เปลี่ยนคะแนน",
            )

            explanation_table["ปัจจัย"] = (
                explanation_table["feature"]
                .map(feature_labels)
                .fillna(
                    explanation_table["feature"]
                )
            )

            explanation_table = (
                explanation_table
                .sort_values(
                    "ความสำคัญ",
                    ascending=False,
                )
                .head(5)
                .copy()
            )

            explanation_table[
                "ผลต่อโมเดล (SHAP)"
            ] = explanation_table[
                "shap_value"
            ].round(4)

            st.dataframe(
                explanation_table[
                    [
                        "ปัจจัย",
                        "ค่าที่ใช้",
                        "ผลต่อโมเดล (SHAP)",
                        "ทิศทาง",
                    ]
                ],
                hide_index=True,
                use_container_width=True,
            )

            st.caption(
                "ค่า SHAP อธิบายว่าปัจจัยใดทำให้"
                "คะแนนของโมเดลเพิ่มหรือลด "
                "แต่ไม่ได้พิสูจน์ความสัมพันธ์"
                "เชิงเหตุและผล"
            )

        except Exception as error:
            st.warning(
                "ไม่สามารถแสดงคำอธิบาย SHAP ได้: "
                f"{error}"
            )

