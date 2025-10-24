import gradio as gr
import pandas as pd
import joblib, json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/random_forest_final.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "../models/feature_columns.json")
CATEGORIES_PATH = os.path.join(BASE_DIR, "../models/categories.json")

model = joblib.load(os.path.normpath(MODEL_PATH))
with open(os.path.normpath(FEATURES_PATH), "r", encoding="utf-8") as f:
    train_cols = json.load(f)
with open(os.path.normpath(CATEGORIES_PATH), "r", encoding="utf-8") as f:
    meta = json.load(f)

cats = meta["choices"]
brand_models = meta["brand_models"]


def estimate_price(brand, model_name, model_year, milage, fuel_type,
                   transmission, ext_col, int_col, accident, clean_title):

    df = pd.DataFrame([{
        "brand": brand,
        "model": model_name,
        "car_age": 2025 - int(model_year),
        "milage": float(milage),
        "fuel_type": fuel_type,
        "transmission": transmission,
        "ext_col": ext_col,
        "int_col": int_col,
        "accident": accident,
        "clean_title": clean_title
    }])

    for col in ["brand","model","fuel_type","transmission","ext_col","int_col","accident","clean_title"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    df_dummies = pd.get_dummies(df, drop_first=False)

    df_dummies = df_dummies.reindex(columns=train_cols, fill_value=0)

    print("Aktive features (sum):", df_dummies.sum(axis=1).values)
    print("Noen kolonner:", [c for c in df_dummies.columns if df_dummies.iloc[0][c] == 1][:10])

    y = model.predict(df_dummies)[0]
    return f"{int(y * 11):,} kr"




def update_model_choices(selected_brand):
    models = brand_models.get(selected_brand, cats["model"])
    return gr.Dropdown(choices=models, value=None)

with gr.Blocks(title="🚗 Bruktbil Pris Estimator") as demo:
    gr.Markdown("### Skriv inn informasjon om bilen din for å estimere markedsverdi")

    with gr.Row():
        brand_dd = gr.Dropdown(choices=cats["brand"], label="Merke", allow_custom_value=True)
        model_dd = gr.Dropdown(choices=cats["model"], label="Modell", allow_custom_value=True)

    brand_dd.change(fn=update_model_choices, inputs=brand_dd, outputs=model_dd)

    with gr.Row():
        year_in = gr.Number(label="Årsmodell", value=2018)
        km_in = gr.Slider(0, 500000, value=120000, step=1000, label="Kjørelengde (km)")

    with gr.Row():
        fuel_dd = gr.Dropdown(choices=cats["fuel_type"], label="Drivstoff", allow_custom_value=True)
        trans_dd = gr.Dropdown(choices=cats["transmission"], label="Girkasse", allow_custom_value=True)

    with gr.Row():
        ext_dd = gr.Dropdown(choices=cats["ext_col"], label="Utvendig farge", allow_custom_value=True)
        int_dd = gr.Dropdown(choices=cats["int_col"], label="Innvendig farge", allow_custom_value=True)

    acc_dd = gr.Radio(
        choices=["None reported", "At least 1 accident or damage reported"],
        value="None reported",
        label="Skaderapport"
    )

    clean_dd = gr.Radio(
        choices=["Yes", "No"],
        value="Yes", 
        label="Ren tittel"
    )

    out = gr.Textbox(label="Estimert pris")

    btn = gr.Button("Estimer pris")
    btn.click(
        fn=estimate_price,
        inputs=[brand_dd, model_dd, year_in, km_in, fuel_dd, trans_dd, ext_dd, int_dd, acc_dd, clean_dd],
        outputs=out
    )

if __name__ == "__main__":
    demo.launch()
