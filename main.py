from src.pipeline import MLPipeline

if __name__ == "__main__":
    pipeline = MLPipeline(
        data_path="data/raw/train.csv",
        target_col="Price",
        scale_cols=[
            "Category", "Fuel type", "Engine volume",
            "Mileage", "Cylinders", "Airbags", "Levy"
        ]
    )

    model, params, metrics = pipeline.run()

    print("Best Params:", params)
    print("Metrics:", metrics)
